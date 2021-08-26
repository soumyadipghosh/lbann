////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/base.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/models/directed_acyclic_graph.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"

#include <training_algorithm.pb.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace lbann {
namespace ltfb {
namespace {

El::Int cycle_winner(std::vector<EvalType> scores, RegularizedEvolution::metric_strategy strategy)
{
  using MetricStrategy = RegularizedEvolution::metric_strategy;
  switch (strategy) {
  case MetricStrategy::LOWER_IS_BETTER:
    return (std::min_element(scores.cbegin(), scores.cend()) - scores.cbegin());
  case MetricStrategy::HIGHER_IS_BETTER:
    return (std::max_element(scores.cbegin(), scores.cend()) - scores.cbegin());
  default:
    LBANN_ERROR("Invalid metric strategy!");
  }
  return false; // Silence compiler warning about no return.
}

// Pack model to ship off
std::string pack(model const& m)
{
  std::ostringstream oss;
  {
    RootedBinaryOutputArchive ar(oss, m.get_comm()->get_trainer_grid());
    ar(m);
  }
  return oss.str();
}

// Send a string to the root of the destination trainer
void send_string(lbann_comm const& comm,
                 std::string const& str,
                 int destination_trainer)
{
  size_t size = str.length();
  comm.send(&size, 1, destination_trainer, /*rank=*/0);
  comm.send(reinterpret_cast<El::byte const*>(str.data()),
            size,
            destination_trainer,
            /*rank=*/0);
}

// Receive a string from the root of src_trainer
std::string recv_string(lbann_comm const& comm, int src_trainer)
{
  size_t size = 0;
  comm.recv(&size, 1, src_trainer);
  std::string buf;
  buf.resize(size);
  comm.recv(reinterpret_cast<El::byte*>(buf.data()), size, src_trainer);
  return buf;
}

// Unpack received model
void unpack(model& m, std::string const& str)
{
  std::istringstream iss(str);
  {
    RootedBinaryInputArchive ar(iss, m.get_comm()->get_trainer_grid());
    ar(m);
  }
}

} // namespace

// RegularizedEvolution Implementation

RegularizedEvolution::RegularizedEvolution(
  std::unordered_map<std::string, metric_strategy> metrics,
  std::unique_ptr<MutationStrategy> mutate_algo)
  : m_metrics{std::move(metrics)},
    m_mutate_algo{std::move(mutate_algo)}
{
  LBANN_ASSERT(m_metrics.size());
}

RegularizedEvolution::RegularizedEvolution(
  std::string metric_name,
  metric_strategy winner_strategy,
  std::unique_ptr<MutationStrategy> mutate_algo)
  : RegularizedEvolution({metric_name, winner_strategy},
                         std::move(mutate_algo))
{}

RegularizedEvolution::RegularizedEvolution(
  RegularizedEvolution const& other)
  : m_metrics{other.m_metrics}, m_mutate_algo{other.m_mutate_algo}
{}

EvalType RegularizedEvolution::evaluate_model(model& m,
                                              ExecutionContext& ctxt,
                                              data_coordinator& dc) const

{
  // Make sure data readers finish asynchronous work
  const auto original_mode = ctxt.get_execution_mode();
  dc.collect_background_data_fetch(original_mode);

  // Can use validation if it is global
  if (!dc.is_execution_mode_valid(execution_mode::tournament)) {
    LBANN_ERROR("Regularized Evolution requires ",
                to_string(execution_mode::tournament),
                " execution mode");
  }

  // Mark the data store as loading - Note that this is a temporary fix
  // for the current use of the tournament
  m.mark_data_store_explicitly_loading(execution_mode::tournament);

  // Evaluate model on test (or validation?) set
  get_trainer().evaluate(&m, execution_mode::tournament);

  // Get metric values
  bool found_metric = false;
  EvalType score = 0.f;
  std::string metric_name;
  for (const auto& met : m.get_metrics()) {
    metric_name = met->name();
    if (m_metrics.count(metric_name)) {
      found_metric = true;
      score += met->get_mean_value(execution_mode::tournament);
      break;
    }
  } 

  // sanity check
  if (!found_metric) {
    LBANN_ERROR("could not find metric \"",
                metric_name,
                "\" "
                "in model \"",
                m.get_name(),
                "\"");
  }

  m.make_data_store_preloaded(execution_mode::testing);

  // Clean up and return metric score
  m.reset_mode(ctxt, original_mode);
  dc.reset_mode(ctxt);
  return score;
}

void RegularizedEvolution::select_next(model& m,
                                       ltfb::ExecutionContext& ctxt,
                                       data_coordinator& dc) const
{
  // Find the best model trainer
  // Find the oldest model trainer (highest age)
  // Copy the best model trainer to oldest model trainer, mutate it and set its age to 0

  auto const& comm = *(m.get_comm());
  const unsigned int num_trainers = comm.get_num_trainers();
  const int trainer_id = comm.get_trainer_rank();
  auto const step = ctxt.get_step();

  // Choose sample S<P
  int S = 10;
  std::vector<EvalType> sample_trainers(num_trainers);
  if (comm.am_world_master()) {
    for (int i = 0; i < num_trainers; i++) sample_trainers[i] = i;
    std::shuffle(sample_trainers.begin(), sample_trainers.end(), get_ltfb_generator());
  }
  comm.world_broadcast(comm.get_world_master(), sample_trainers.data(), num_trainers);

  // if rank within first S, send score , else score = 0
  auto it = std::find(sample_trainers.begin(), sample_trainers.end(), trainer_id);
  
  El::Int score;
  // If in sample, send true score
  if (std::distance(sample_trainers.begin(), it) < S) {
    score = evaluate_model(m, ctxt, dc);
  }
  // Else send dummy score
  else {
    score = 0;
  }  
 
  std::vector<EvalType> score_list(num_trainers);
  comm.trainer_barrier();
  if (comm.am_trainer_master()) {
    comm.all_gather<EvalType>(score, score_list, comm.get_intertrainer_comm());
  }
  // Communicate trainer score list from trainer master to other procs in trainer
  comm.trainer_broadcast(comm.get_trainer_master(), score_list.data(), num_trainers);
  
  // Find winning trainer (in sample)
  El::Int winner_id = std::distance(score_list.begin(), 
                         std::max_element(score_list.begin(), score_list.end()));

  // Find oldest trainer - cycle through trainer ids
  El::Int oldest_id = step % num_trainers;

  if (trainer_id == winner_id) {
    auto model_string = pack(m);
    if (comm.am_trainer_master()) {
      send_string(comm, model_string, oldest_id);
      std::cout << "In Reg Evo step " << step << ", trainer " << trainer_id
                << " with score " << score_list[trainer_id];
      std::cout << " sends model to trainer " << oldest_id << std::endl;
    }
  }

  if (trainer_id == oldest_id) {

    std::string rcv_str;
    if (comm.am_trainer_master()) {
      rcv_str = recv_string(comm, src);
      std::cout << "In Reg Evo step " << step << ", trainer " << trainer_id;
      std::cout << " receives model from trainer " << src << std::endl;
    }   

    auto partner_model_ptr = m.copy_model();
    auto& partner_model = *partner_model_ptr;
    unpack(partner_model, rcv_str);

    // Mutating oldest model
    m_mutate_algo->mutate(partner_model, step);

    auto& trainer = get_trainer();
    auto&& metadata = trainer.get_data_coordinator().get_dr_metadata();
    m.setup(trainer.get_max_mini_batch_size(),
            metadata,
            /*force*/true);
  }
}

} // namespace ltfb
} // namespace lbann

