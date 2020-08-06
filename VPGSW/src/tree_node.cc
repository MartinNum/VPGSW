// Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "n2/tree_node.h"

namespace n2 {

TreeNode::TreeNode(int navigation_id, std::vector<int>& sub_nodes, int size)
: navigation_id_(navigation_id), sub_nodes_(sub_nodes), tree_size_(size) {}
TreeNode::TreeNode()
: tree_size_(0){}

void TreeNode::SetNavigationId(int navigation_id){
    navigation_id_ = navigation_id;
}

void TreeNode::SetTreeSize(int size){
    tree_size_ = size;
}

void TreeNode::SetSubNodes(std::vector<int>& sub_nodes){
    sub_nodes_ = sub_nodes;
}

void TreeNode::SetRadius(float radius){
    radius_ = radius;
}

void TreeNode::SetEnterPointId(int enter_point_id){
    enter_point_id_ = enter_point_id;
}

float TreeNode::GetRadius(){
    return radius_;
}

void TreeNode::CopyDataToOptIndex(char* mem_offset, int id) const {
    char* mem_data = mem_offset;
    *((int*)(mem_data)) = (int)(id);
    mem_data += sizeof(int);
    *((int*)(mem_data)) = (int)(navigation_id_);
    mem_data += sizeof(int);
    *((float*)(mem_data)) = (float)(radius_);
    mem_data += sizeof(float);
    *((int*)(mem_data)) = (int)(enter_point_id_);
}

// void HnswNode::CopyLinksToOptIndex(char* mem_offset) const {
//     char* mem_data = mem_offset;
//     const auto& neighbors = friends;
//     *((int*)(mem_data)) = (int)(neighbors.size());
//     mem_data += sizeof(int);
//     for (size_t i = 0; i < neighbors.size(); ++i) {
//         *((int*)(mem_data)) = (int)neighbors[i]->GetId();
//         mem_data += sizeof(int);
//     }
// }

} // namespace n2
