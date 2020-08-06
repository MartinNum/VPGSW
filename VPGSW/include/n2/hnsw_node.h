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

namespace n2 {

class HnswNode {
public:
    explicit HnswNode(int id, const Data* data, int maxsize, int ml);
    void CopyDataAndLinksToOptIndex(char* mem_offset) const;

    inline int GetId() const { return id_; }
    
    inline float GetRadius() { return radius_; }
    inline void SetRadius( float radius ) { radius_ = radius; }

    inline void SetLetfFriends(std::vector<HnswNode*>& new_friends) {
        left_friends.swap(new_friends);
    }

    inline void SetRightFriends(std::vector<HnswNode*>& new_friends) {
        right_friends.swap(new_friends);
    }

    inline void SetDistantFriends(std::vector<HnswNode*>& new_friends){distant_friends.swap(new_friends);}

    inline const std::vector<HnswNode*>& GetLetfFriends() const { return left_friends; }

    inline const std::vector<HnswNode*>& GetRightFriends() const { return right_friends; }

    inline const std::vector<HnswNode*>& GetDistantFriends() const {return distant_friends;}

    inline const std::vector<float>& GetData() const { return data_->GetData(); }
    inline const std::vector<HnswNode*>& GetFriends() const { return friends; }

    inline void SetFriends(std::vector<HnswNode*>& new_friends) {
            friends.swap(new_friends);
    }

private:
    void CopyLinksToOptIndex(char* mem_offset) const;

public:
    int id_;
    float radius_;
    const Data* data_;
    size_t maxsize_;
    size_t ML_;
    std::vector<HnswNode*> friends;
    std::vector<HnswNode*> distant_friends;
    std::vector<HnswNode*> left_friends;
    std::vector<HnswNode*> right_friends;
    
    
    std::mutex access_guard_;
};

} // namespace n2
