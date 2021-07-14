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
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/softmax.hpp"

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
  virtual void mutate(model& m) = 0;
};

// No Mutation
class NullMutation : public Cloneable<NullMutation, MutationStrategy>
{

public:
  NullMutation() = default; 
  void mutate(model&) override {}
};

// Replace activation layers
class ReplaceActivation : public Cloneable<ReplaceActivation, MutationStrategy>
{
private:
  std::string m_old_layer_type;
  std::string m_new_layer_type;

public:
  //ReplaceActivation() = default;
  ReplaceActivation(std::string const& old_layer_type, std::string const& new_layer_type)
         : m_old_layer_type{old_layer_type}, m_new_layer_type{new_layer_type}
  {
    // Convert to lower case for eaasy comparison
    std::transform(m_old_layer_type.begin(), m_old_layer_type.end(), 
                                             m_old_layer_type.begin(), ::tolower);
    std::transform(m_new_layer_type.begin(), m_new_layer_type.end(), 
                                             m_new_layer_type.begin(), ::tolower);
  }

  std::unique_ptr<lbann::Layer> make_new_activation_layer(lbann::lbann_comm& comm,
                                                          std::string const& name)
  {
    std::unique_ptr<lbann::Layer> layer;

    if(m_new_layer_type == "relu") {
       layer = std::make_unique<
           lbann::relu_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(&comm); 
    } else if (m_new_layer_type == "tanh") {
       layer = std::make_unique<
            lbann::tanh_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(&comm);
    } /* else if (m_new_layer_type == "softmax") {
       layer = std::make_unique<
            lbann::softmax_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(&comm);
    }*/ else {
       LBANN_ERROR("Unknown new layer type: ", m_new_layer_type);
    }
    layer->set_name(name);
    return layer;
  }

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
    std::cout << "hello from replace" << std::endl;

    auto& comm = *m.get_comm();

    std::vector<std::string> all_old_layer_type_names;

    // Find all old layer type names
    for (int i=0; i<m.get_num_layers(); ++i) {
       // Creating a temp string with lower case representation
       std::string temp_type = m.get_layer(i).get_type();
       std::transform(temp_type.begin(), temp_type.end(),
                                         temp_type.begin(), ::tolower);    

       if (temp_type == m_old_layer_type) {
         all_old_layer_type_names.push_back(m.get_layer(i).get_name());
       }
    }

    // Replace them with new layer type
    for (size_t i=0UL; i<all_old_layer_type_names.size(); i++) {
       // new layer name
       std::string new_layer_name = m_new_layer_type + "_" + std::to_string(i);

       // Call replace for each of them
       m.replace_layer(make_new_activation_layer(comm, new_layer_name), all_old_layer_type_names[i]);
    }
  }
};

} // namespace ltfb

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
