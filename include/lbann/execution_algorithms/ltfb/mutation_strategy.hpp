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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED

#include "lbann/models/model.hpp"
#include "lbann/layers/math/unary.hpp"

// What else ?

namespace lbann {
namespace ltfb {

class MutationStrategy : public Cloneable<HasAbstractFunction<MutationStrategy>>
{
public:
  MutationStrategy() {};
  virtual ~MutationStrategy() = default;

public:
  /** @brief Apply a change to the model.
   *  @param[in,out] m The model to change.
   */
  virtual void mutate(model& m) const = 0;
};

// No Mutation
class NullMutation : public Cloneable<NullMutation, MutationStrategy>
{

public:
  NullMutation() = default; 
  void mutate(model&) const override {}
};

// Replace activation layers
class ReplaceActivation : public Cloneable<ReplaceActivation, MutationStrategy>
{

public:
  ReplaceActivation() = default;

  std::unique_ptr<lbann::Layer> make_new_tanh_layer(lbann::lbann_comm& comm, std::string const& name)
  {
    auto layer = std::make_unique<
      lbann::tanh_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(
      &comm);
    layer->set_name(name);
    return layer;
  }

  void mutate(model& m)
  {
    auto& comm = *m.get_comm();

    std::vector<std::string> all_relu;

    // Find all relu
    for (int i=0; i<m.get_num_layers(); ++i) {
       if (m.get_layer(i).get_type() == "relu") {
         all_relu.push_back(m.get_layer(i).get_name());
       }
    }

    // Replace them with tanh
    for (size_t i=0UL; i<all_relu.size(); i++) {
       // make tanh with appropriate name
       std::string new_tanh = "new_tanh" + std::to_string(i);

       // Call replace for each of them
       m.replace_layer(make_new_tanh_layer(comm, new_tanh), all_relu[i]);
    }

    // MODEL WILL BE SETUP IN RPE - NO NEED TO DO HERE
  }
};

} // namespace ltfb

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
