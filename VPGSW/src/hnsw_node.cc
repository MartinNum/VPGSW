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

#include "n2/hnsw_node.h"

namespace n2 {

HnswNode::HnswNode(int id, const Data* data, int maxsize, int ml)
: id_(id), data_(data), maxsize_(maxsize), ML_(ml) {
}

void HnswNode::CopyDataAndLinksToOptIndex(char* mem_offset) const {
    char* mem_data = mem_offset;
    *((float*)(mem_data)) = (float)(radius_);
    mem_data += sizeof(float);
    CopyLinksToOptIndex(mem_data);
    mem_data += (sizeof(int) * 2 + sizeof(int)*(maxsize_ + ML_));
    auto& data = data_->GetData();
    for (size_t i = 0; i < data.size(); ++i) {
        *((float*)(mem_data)) = (float)data[i];
        mem_data += sizeof(float);
    }
}

void HnswNode::CopyLinksToOptIndex(char* mem_offset) const {
    char* mem_data = mem_offset;
    const auto& letf_neighbors = left_friends;
    *((int*)(mem_data)) = (int)(letf_neighbors.size());
    mem_data += sizeof(int);
    for (size_t i = 0; i < letf_neighbors.size(); ++i) {
        *((int*)(mem_data)) = (int)letf_neighbors[i]->GetId();
        mem_data += sizeof(int);
    }

    const auto& right_neighbors = right_friends;
    *((int*)(mem_data)) = (int)(right_neighbors.size());
    mem_data += sizeof(int);
    for (size_t i = 0; i < right_neighbors.size(); ++i) {
        *((int*)(mem_data)) = (int)right_neighbors[i]->GetId();
        mem_data += sizeof(int);
    }
}

} // namespace n2
