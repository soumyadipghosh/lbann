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

#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/convolution.hpp"

#include "lbann/layers/transform/reshape.hpp"

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
    // Convert to lower case for easy comparison
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

// Replace Learnable Layers
class ReplaceLearnable : public Cloneable<ReplaceLearnable, MutationStrategy>
{
public:
  ReplaceLearnable() = default;

  std::unique_ptr<lbann::Layer> make_new_learnable_layer(lbann::lbann_comm& comm,
                                                  std::string const& name)
  {
    auto layer = std::make_unique<
      lbann::fully_connected_layer<float, data_layout::DATA_PARALLEL, El::Device::GPU>>
                                  (120, false, nullptr, true);
    /*
    std::vector<int> layer_dims{5,5}, layer_pads{0,0}, layer_strides{1,1}, layer_dilations{1,1};
    auto layer = std::make_unique<
      lbann::convolution_layer<float, data_layout::DATA_PARALLEL, El::Device::GPU>>
                            (2, 6, layer_dims, layer_pads, layer_strides, layer_dilations, 1, true);
    */
    layer->set_name(name);
    return layer;
  }

  void mutate(model& m)
  {
    auto& comm = *m.get_comm();

    std::vector<std::string> learnable_layer_names;

    // Find the first fully connected layer and replace it with a new one
    for (auto i=0; i<m.get_num_layers(); ++i) {
       // Creating a temp string with lower case representation
       std::string temp_type = m.get_layer(i).get_type();
       std::transform(temp_type.begin(), temp_type.end(),
                                         temp_type.begin(), ::tolower);  

       /*
       // Print output dims of all layers
       std::vector<int> od = m.get_layer(i).get_output_dims(0);
       int sz = m.get_layer(i).get_output_size(0);
       std::cout << " Size of layer - " << sz << std::endl;
       std::cout << " Dims of layer - ";
       for (size_t j=0UL; j < od.size(); j++) std::cout << od[j] << " ";
       std::cout << std::endl;
       */
       
       if (temp_type == "fully connected") {
       // if (temp_type == "convolution") {
         learnable_layer_names.push_back(m.get_layer(i).get_name());
         //break; // just the first fc for now
       }   
     }

     // Replace with new layers
     std::string new_layer_name = "new_fc";
     //std::string new_layer_name = "new_conv";

     // just first layer
     std::cout << "Attempting to replace " << learnable_layer_names[0] << std::endl;
     // Call replace for each of them
     m.replace_layer(make_new_learnable_layer(comm, new_layer_name), learnable_layer_names[0]);
  }
};

// Insert Conv layers - assume we insert a 3rd conv layer in Lenet
class InsertConv : public Cloneable<InsertConv, MutationStrategy>
{
public:
  InsertConv() = default;

  std::unique_ptr<lbann::Layer> make_new_conv_layer(lbann::lbann_comm& comm, std::string const& name)
  {
    std::vector<int> layer_dims{5,5}, layer_pads{0,0}, layer_strides{1,1}, layer_dilations{1,1};
    auto layer = std::make_unique<
      lbann::convolution_layer<float, data_layout::DATA_PARALLEL, El::Device::GPU>>
                            (2, 16, layer_dims, layer_pads, layer_strides, layer_dilations, 1, true);
    layer->set_name(name);
    return layer;
  }

  std::unique_ptr<lbann::Layer> make_new_fc_layer(lbann::lbann_comm& comm, std::string const& name)
  {
    auto layer = std::make_unique<
      lbann::fully_connected_layer<float, data_layout::DATA_PARALLEL, El::Device::GPU>>
                                  (4056, false, nullptr, true);
    layer->set_name(name);
    return layer;
  }

  std::unique_ptr<lbann::Layer> make_new_relu_layer(lbann::lbann_comm& comm, std::string const& name)
  {
    auto layer = std::make_unique<
      lbann::relu_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(
      &comm);    
    layer->set_name(name);
    return layer;
  }

  std::unique_ptr<lbann::Layer> make_new_reshape_layer(lbann::lbann_comm& comm,
                                                       std::string const& name, std::vector<int> dims)
  {
    auto layer = std::make_unique<
      lbann::reshape_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(&comm, dims);
    layer->set_name(name);
    return layer;
  }
 
  void mutate(model& m)
  {
    auto& comm = *m.get_comm();
    //auto& trainer = lbann::get_trainer();
    //auto&& metadata = trainer.get_data_coordinator().get_dr_metadata();

    std::vector<std::string> pooling_layer_names;   
    std::vector<std::string> conv_layer_names; 
    std::vector<int> conv_layer_indices;

    // Find the names of the pooling layers
    for (auto i=0; i<m.get_num_layers(); ++i) {
       // Creating a temp string with lower case representation
       std::string temp_type = m.get_layer(i).get_type();
       std::transform(temp_type.begin(), temp_type.end(),
                                         temp_type.begin(), ::tolower);

       if (temp_type == "pooling") {
         pooling_layer_names.push_back(m.get_layer(i).get_name());
       }

       if (temp_type == "convolution") {
         conv_layer_names.push_back(m.get_layer(i).get_name());
         conv_layer_indices.push_back(i);
       }
    }

    auto prev_pool = pooling_layer_names[0];  
    auto next_conv = conv_layer_names[1];
    auto conv_dims = m.get_layer(conv_layer_indices[1]).get_input_dims(0);

    std::string new_layer_name = "new_conv";

    m.insert_layer(make_new_conv_layer(comm, new_layer_name), prev_pool);              
    //m.setup(trainer.get_max_mini_batch_size(), metadata, true);

    m.insert_layer(make_new_relu_layer(comm, "new_relu"), new_layer_name);

    // Check and insert shimming FC layers before and after
    m.insert_layer(make_new_fc_layer(comm, "new_fc"), "new_relu");

    m.insert_layer(make_new_relu_layer(comm, "new_relu2"), "new_fc");

    m.insert_layer(make_new_reshape_layer(comm, "new_reshape", conv_dims), "new_relu2");

    /*
    // Print dimensions
    for (auto i=1; i<m.get_num_layers(); ++i) {
       std::vector<int> id = m.get_layer(i).get_input_dims(0);
       std::vector<int> od = m.get_layer(i).get_output_dims(0);
       int isz = m.get_layer(i).get_input_size(0);
       int osz = m.get_layer(i).get_output_size(0);

       std::cout << " I/P " << i << " : size - " << isz;
       std::cout << " dims - ";
       for (size_t j=0UL; j < id.size(); j++) std::cout << id[j] << " ";
       std::cout << std::endl;

       std::cout << " O/P " << i << " : size - " << osz;
       std::cout << " dims - ";
       for (size_t j=0UL; j < od.size(); j++) std::cout << od[j] << " ";
       std::cout << std::endl;
    }
    */
    
  }
};

// Replace Kernel in Conv
class ReplaceKernelConv : public Cloneable<ReplaceKernelConv, MutationStrategy>
{

public:
  ReplaceKernelConv() = default;

  /*
  std::unique_ptr<lbann::Layer> make_new_conv_layer(lbann::lbann_comm& comm, int const& kernel,
                                                    int const& channels, std::string const& name)
  {
    int diff = kernel - 2; // (new - (old - 1)); since old kernel is 3 for both conv layers, (3-1) is hard-coded
    std::vector<int> layer_dims{kernel, kernel}, layer_pads{(diff - 1)/2, (diff - 1)/2},
                     layer_strides{1,1}, layer_dilations{1,1};
    auto layer = std::make_unique<
      lbann::convolution_layer<float, data_layout::DATA_PARALLEL, El::Device::GPU>>
                            (2, channels, layer_dims, layer_pads, layer_strides, layer_dilations, 1, true);
    layer->set_name(name);
    return layer;
  }
  */

  std::unique_ptr<lbann::Layer> make_new_conv_layer(lbann::lbann_comm& comm, int const& kernel,
                                                    int const& channels, std::string const& name)
  {
    std::vector<int> layer_dims{kernel, kernel}, layer_pads{0, 0}, layer_strides{1, 1}, layer_dilations{1, 1};
    auto layer = std::make_unique<
      lbann::convolution_layer<float, data_layout::DATA_PARALLEL, El::Device::GPU>>
                            (2, channels, layer_dims, layer_pads, layer_strides, layer_dilations, 1, true);
    layer->set_name(name);
    return layer;
  }

  // Find all conv layers and replace them with a conv layer of different kernel and suitable padding
  void mutate(model& m)
  {
    auto& comm = *m.get_comm();

    std::vector<int> kernel_sizes = {7}; // using just one kernel size for now
    std::vector<int> channels = {10, 12};
    std::vector<int> conv_layer_indices; // indices of all conv layers

    for (auto i = 0; i < m.get_num_layers(); ++i) {
       // Creating a temp string with lower case representation
       std::string temp_type = m.get_layer(i).get_type();
       std::transform(temp_type.begin(), temp_type.end(),
                                         temp_type.begin(), ::tolower); 
      
       std::string temp_name = m.get_layer(i).get_name();
       // Ensure that conv shim layers are not counted here

       if (temp_type == "convolution" && (temp_name.find("shim") == std::string::npos)) {
         conv_layer_indices.push_back(i);
       }
    }

    /* // For just kernel
    int conv_index;
    int kernel_index;
    if (m.get_comm()->am_trainer_master()) {
      conv_index = fast_rand_int(get_fast_generator(), conv_layer_indices.size());
      kernel_index = fast_rand_int(get_fast_generator(), kernel_sizes.size());
    }
    m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(), conv_index);
    m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(), kernel_index);

    auto& layer = m.get_layer(conv_layer_indices[conv_index]);
    std::string old_name = layer.get_name();
    std::string new_name = old_name;
    auto channels = layer.get_output_dims(0)[0];

    std::cout << "Changing kernel size in " << old_name << " to " << kernel_sizes[kernel_index]
                                   << " in trainer " << m.get_comm()->get_trainer_rank() << std::endl;

    m.replace_layer(make_new_conv_layer(comm, kernel_sizes[kernel_index], channels, new_name),
                    old_name);   
    */

    int conv_index;
    int channel_index;
    if (m.get_comm()->am_trainer_master()) {
      conv_index = fast_rand_int(get_fast_generator(), conv_layer_indices.size());
      channel_index = fast_rand_int(get_fast_generator(), channels.size());
    }
    m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(), conv_index);
    m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(), channel_index);

    auto& layer = m.get_layer(conv_layer_indices[conv_index]);
    std::string old_name = layer.get_name();
    std::string new_name = old_name;
    auto old_channels = layer.get_output_dims(0)[0];
    using base_conv = lbann::base_convolution_layer<DataType, El::Device::GPU>;
    auto& cast_layer = dynamic_cast<base_conv&>(layer);
    auto kernel = cast_layer.get_conv_dims()[0];

    std::cout << "Changing channels in " << old_name << " from " << old_channels
              << " to " << channels[channel_index] << std::endl;

    // Store the child of the conv layer before replacing it
    auto& child = layer.get_child_layer(0);
   
    // Replace the channels in the conv layer
    m.replace_layer(make_new_conv_layer(comm, kernel, channels[channel_index], new_name),
                    old_name);
 
    // Find out if there is a shim layer. If not, insert a shim 1x1 conv layer to match channels
    if (child.get_name().find("shim") != std::string::npos) { // if already there

      std::cout << "Replacing shim" << std::endl;
      
      std::string shim_layer_name = child.get_name();

      // Find child of shim layer
      auto& child_of_shim = child.get_child_layer(0);
      auto child_of_shim_channels = child_of_shim.get_input_dims(0)[0];

      //replace shim layer; o/p channels stay same but i/p channels change
      m.remove_layer(shim_layer_name);
      m.insert_layer(make_new_conv_layer(comm, 1, child_of_shim_channels, shim_layer_name),
                     new_name); 
      //m.replace_layer(make_new_conv_layer(comm, 1, child_of_shim_channels, shim_layer_name),
      //                shim_layer_name);

    }
    else { // if not, create a new shim layer. Should enter this only once

      std::cout << "Creating shim" << std::endl;

      // old_channels should be equal to child_of_shim_channels since
      // this block should get executed only once
      std::string shim_layer_name = new_name + "_shim";
      m.insert_layer(make_new_conv_layer(comm, 1, old_channels, shim_layer_name),
                     new_name);  
    }                     
  }    
};

} // namespace ltfb

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
