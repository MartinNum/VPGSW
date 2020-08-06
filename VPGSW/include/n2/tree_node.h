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

#pragma once

#include <vector>
#include <mutex>

#include "base.h"
#include "hnsw_node.h"

namespace n2 {

class TreeNode {
public:
    explicit TreeNode(int navigation_id, std::vector<int>& sub_nodes, int size);
    explicit TreeNode();

    void SetNavigationId(int navigation_id);
    void SetTreeSize(int size);
    void SetSubNodes(std::vector<int>& sub_nodes);
    void SetRadius(float radius);
    void SetEnterPointId(int enter_point_id);
    float GetRadius();
    void CopyDataToOptIndex(char* mem_offset, int id) const;

    // inline int GetId() const { return id_; }
    // inline const std::vector<float>& GetData() const { return data_->GetData(); }
    // inline const std::vector<HnswNode*>& GetFriends() const { return friends; }

    // inline void SetFriends(std::vector<HnswNode*>& new_friends) {
    //         friends.swap(new_friends);
    // }

private:
    // void CopyLinksToOptIndex(char* mem_offset) const;

public:
    // int id_;
    // const Data* data_;
    int navigation_id_;
    float radius_;
    int enter_point_id_;
    TreeNode* right_treenode_;
    TreeNode* left_treenode_;

    size_t tree_size_;
    std::vector<int> sub_nodes_;
    
    std::mutex access_guard_;
};

} // namespace n2
