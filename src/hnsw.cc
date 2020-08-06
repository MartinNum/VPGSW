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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <xmmintrin.h>

#include "n2/hnsw.h"
#include "n2/hnsw_node.h"
#include "n2/distance.h"
#include "n2/min_heap.h"
#include "n2/tree_node.h"

#define MERGE_BUFFER_ALGO_SWITCH_THRESHOLD 100

namespace n2 {

using std::endl;
using std::fstream;
using std::max;
using std::min;
using std::mutex;
using std::ofstream;
using std::ifstream;
using std::pair;
using std::priority_queue;
using std::setprecision;
using std::string;
using std::stof;
using std::stoi;
using std::to_string;
using std::unique_lock;
using std::unordered_set;
using std::vector;


thread_local VisitedList* visited_list_ = nullptr;

Hnsw::Hnsw() {
    logger_ = spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    metric_ = DistanceKind::ANGULAR;
    dist_cls_ = new AngularDistance();
}

Hnsw::Hnsw(int dim, string metric) : data_dim_(dim) {
    logger_ = spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    if (metric == "L2" || metric =="euclidean") {
        metric_ = DistanceKind::L2;
        dist_cls_ = new L2Distance();
    } else if (metric == "angular") {
        metric_ = DistanceKind::ANGULAR;
        dist_cls_ = new AngularDistance();
    } else {
        throw std::runtime_error("[Error] Invalid configuration value for DistanceMethod: " + metric);
    }
}

Hnsw::Hnsw(const Hnsw& other) {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    model_byte_size_ = other.model_byte_size_;
    model_ = new char[model_byte_size_];
    std::copy(other.model_, other.model_ + model_byte_size_, model_);
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
}

Hnsw::Hnsw(Hnsw& other) {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    model_byte_size_ = other.model_byte_size_;
    model_ = new char[model_byte_size_];
    std::copy(other.model_, other.model_ + model_byte_size_, model_);
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
}

Hnsw::Hnsw(Hnsw&& other) noexcept {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    model_byte_size_ = other.model_byte_size_;
    model_ = other.model_;
    other.model_ = nullptr;
    model_mmap_ = other.model_mmap_;
    other.model_mmap_ = nullptr;
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
}

Hnsw& Hnsw::operator=(const Hnsw& other) {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }

    if(model_) {
        delete [] model_;
        model_ = nullptr;
    }

    if(dist_cls_) {
       delete dist_cls_;
       dist_cls_ = nullptr;
    }

    model_byte_size_ = other.model_byte_size_;
    model_ = new char[model_byte_size_];
    std::copy(other.model_, other.model_ + model_byte_size_, model_);
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
    return *this;
}

Hnsw& Hnsw::operator=(Hnsw&& other) noexcept {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    if(model_mmap_) {
        delete model_mmap_;
        model_mmap_ = nullptr;
    } else {
        delete [] model_;
        model_ = nullptr;
    }

    if(dist_cls_) {
       delete dist_cls_;
       dist_cls_ = nullptr;
    }

    model_byte_size_ = other.model_byte_size_;
    model_ = other.model_;
    other.model_ = nullptr;
    model_mmap_ = other.model_mmap_;
    other.model_mmap_ = nullptr;
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
    return *this;
}

Hnsw::~Hnsw() {
    if (model_mmap_) {
        delete model_mmap_;
    } else {
        if (model_)
            delete[] model_;
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        delete nodes_[i];
    }

    if (default_rng_) {
        delete default_rng_;
    }

    if (dist_cls_) {
        delete dist_cls_;
    }

    if (selecting_policy_cls_) {
        delete selecting_policy_cls_;
    }

    if (post_policy_cls_) {
        delete post_policy_cls_;
    }
}

int Hnsw::GetComparisonTimes(){
    int temp = comparison_times_;
    comparison_times_ = 0;
    return temp;
}


void Hnsw::SetConfigs(const vector<pair<string, string> >& configs) {
    bool is_levelmult_set = false;
    for (const auto& c : configs) {
        if (c.first == "M") {
            MaxM_ = M_ = (size_t)stoi(c.second);
        } else if (c.first == "efConstruction") {
            efConstruction_ = (size_t)stoi(c.second);
        } else if (c.first == "NumThread") {
            num_threads_ = stoi(c.second);
        } else if (c.first == "Mult") {
            levelmult_ = stof(c.second);
            is_levelmult_set = true;
        } else if (c.first == "NeighborSelecting") {

            if(selecting_policy_cls_) delete selecting_policy_cls_;

            if (c.second == "heuristic") {
                selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
                is_naive_ = false;
            } else if (c.second == "heuristic_save_remains") {
                selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
                is_naive_ = false;
            } else if (c.second == "naive") {
                selecting_policy_cls_ = new NaiveNeighborSelectingPolicies();
                is_naive_ = true;
            } else {
                throw std::runtime_error("[Error] Invalid configuration value for NeighborSelecting: " + c.second);
            }
        } else if (c.first == "GraphMerging") {
            if (c.second == "skip") {
                post_ = GraphPostProcessing::SKIP;
            } else if (c.second == "merge_level0") {
                post_ = GraphPostProcessing::MERGE_LEVEL0;
            } else {
                throw std::runtime_error("[Error] Invalid configuration value for GraphMerging: " + c.second);
            }
        } else if (c.first == "EnsureK") {
            if (c.second == "true") {
                ensure_k_ = true;
            } else {
                ensure_k_ = false;
            }
        } else {
            throw std::runtime_error("[Error] Invalid configuration key: " + c.first);
        }
    }
    if (!is_levelmult_set) {
        levelmult_ = 1 / log(1.0*M_);
    }
}

int Hnsw::DrawLevel(bool use_default_rng) {
    double r = use_default_rng ? uniform_distribution_(*default_rng_) : uniform_distribution_(rng_);
    if (r < std::numeric_limits<double>::epsilon())
        r = 1.0;
    return (int)(-log(r) * levelmult_);
}

void Hnsw::Build(int M, int ef_construction, int n_threads, float mult, NeighborSelectingPolicy neighbor_selecting, GraphPostProcessing graph_merging, bool ensure_k) {
    if ( M > 0 ) MaxM_ = M_ = M;
    if ( ef_construction > 0 ) efConstruction_ = ef_construction;
    if ( n_threads > 0 ) num_threads_ = n_threads;
    levelmult_ = mult > 0 ? mult : 1 / log(1.0*M_);

    if (selecting_policy_cls_) delete selecting_policy_cls_;
    if (neighbor_selecting == NeighborSelectingPolicy::HEURISTIC) {
        selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
        is_naive_ = false;
    } else if (neighbor_selecting == NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS) {
        selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
        is_naive_ = false;
    } else if (neighbor_selecting == NeighborSelectingPolicy::NAIVE) {
        selecting_policy_cls_ = new NaiveNeighborSelectingPolicies();
        is_naive_ = true;
    }
    post_ = graph_merging;
    ensure_k_ = ensure_k;
    Fit();
}

void Hnsw::Fit() {
    if (data_.size() == 0) throw std::runtime_error("[Error] No data to fit. Load data first.");
    if (default_rng_ == nullptr)
        default_rng_ = new std::default_random_engine(100);
    rng_.seed(rng_seed_);
    BuildGraph(false);
    if (post_ == GraphPostProcessing::MERGE_LEVEL0) {
        vector<HnswNode*> nodes_backup;
        nodes_backup.swap(nodes_);
        BuildGraph(true);
        MergeEdgesOfTwoGraphs(nodes_backup);
        for (size_t i = 0; i < nodes_backup.size(); ++i) {
            delete nodes_backup[i];
        }
        nodes_backup.clear();
    }

    enterpoint_id_ = enterpoint_->GetId();
    num_nodes_ = nodes_.size();
    long long model_config_size = GetModelConfigSize();
    memory_per_data_ = sizeof(float) * data_dim_;
    memory_per_link_ = sizeof(int) * (2 + MaxM_ + ML_) + sizeof(float);  // "1" for offset pos, 1" for saving num_links
    memory_per_node_ = memory_per_link_ + memory_per_data_;
    memory_per_tree_node_ = sizeof(int)*3 + sizeof(float);
    long long all_node_size = memory_per_node_ * data_.size();
    long long all_tree_node_size = memory_per_tree_node_ * ((1<<Depth_));

    model_byte_size_ = model_config_size + all_node_size + all_tree_node_size;
    model_ = new char[model_byte_size_];
    if (model_ == NULL) {
        throw std::runtime_error("[Error] Fail to allocate memory for optimised index (size: "
                                 + to_string(model_byte_size_ / (1024 * 1024)) + " MBytes)");
    }
    memset(model_, 0, model_byte_size_);
    model_node0_ = model_ + model_config_size;
    model_tree_node0_ = model_node0_ + all_node_size;

    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    float PORTABLE_ALIGN32 TmpRes[8];

    for(int i = 0; i < nodes_.size(); i++){
        std::vector<HnswNode*>& neighbors = nodes_[i]->friends;
        std::vector<HnswNode*>& distant_neighbors = nodes_[i]->distant_friends;

        priority_queue<CloserFirstNew> res;
        for(int j=0; j < neighbors.size(); j++){
            float d = dist_cls_->Evaluate((float*)&(nodes_[i]->GetData()[0]), (float*)&(neighbors[j]->GetData()[0]), data_dim_, TmpRes);
            res.emplace(neighbors[j]->GetId(), d);
        }
        for(int j=0; j < distant_neighbors.size(); j++){
            float d = dist_cls_->Evaluate((float*)&(nodes_[i]->GetData()[0]), (float*)&(distant_neighbors[j]->GetData()[0]), data_dim_, TmpRes);
            res.emplace(distant_neighbors[j]->GetId(), d);
        }
        std::cout<<endl;

        std::vector<HnswNode*>& left_neighbors = nodes_[i]->left_friends;
        std::vector<HnswNode*>& right_neighbors = nodes_[i]->right_friends;

        for(int j=0; j < (neighbors.size()+distant_neighbors.size())/2; j++){
            // std::cout<<res.top().GetDistance()<<"  ";
            left_neighbors.push_back(nodes_[res.top().GetId()]);
            res.pop();
        }

        nodes_[i]->SetRadius(res.top().GetDistance());
        // std::cout<<res.top().GetDistance()<<"  "<<endl;
        // for(int j = 0;j < ((neighbors.size()+distant_neighbors.size()) - (neighbors.size()+distant_neighbors.size())/2); j++){
        //     // std::cout<<res.top().GetDistance()<<"  ";
        //     right_neighbors.push_back(nodes_[res.top().GetId()]);
        //     res.pop();
        // }
        while(!res.empty()){
            right_neighbors.push_back(nodes_[res.top().GetId()]);
            res.pop();
        }
    }

    // for(int i = 0;i < nodes_.size(); i++){

    //     std::vector<HnswNode*>& neighbors = nodes_[i]->friends;
    //     std::cout<<"节点"<<i<<": ";
    //     for(int j=0; j < neighbors.size(); j++){
    //         float d = dist_cls_->Evaluate((float*)&(nodes_[i]->GetData()[0]), (float*)&(neighbors[j]->GetData()[0]), data_dim_, TmpRes);
    //         std::cout<<d<<"  ";
    //     }
    //     std::cout<<endl;
    // }

    SaveModelConfig(model_);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        nodes_[i]->CopyDataAndLinksToOptIndex(model_node0_ + i * memory_per_node_);
    }

    std::vector<int> sub_node;
    sub_node.resize(nodes_.size());
    for(int i=0; i < nodes_.size(); i++){
        sub_node[i] = nodes_[i]->GetId();
    }
    TreeNode* root_tree_node = createCompleteTree(1, sub_node);
    std::queue<TreeNode*> q;
    q.push(root_tree_node);
    int id = 0;
    while(!q.empty()){
        TreeNode* tree_node = q.front();
        q.pop();
        id += 1;
        tree_node->CopyDataToOptIndex(model_tree_node0_ + (id-1)*memory_per_tree_node_, id);

        // if(id < 1039){
        //     std::cout<<"tree_node_id = "<<id<<"   ";
        //     std::cout<<"size = "<<tree_node->tree_size_<<"  ";
        //     std::cout<<"radius = "<<tree_node->radius_<<endl;
        // }

        // std::cout<<"tree_node_id = "<<id<<"   ";
        // std::cout<<"size = "<<tree_node->tree_size_<<"  ";
        // std::cout<<"radius = "<<tree_node->radius_<<endl;
        // std::cout<<"navigation_id = "<<tree_node->navigation_id_<<endl;

        if(tree_node->GetRadius()!=0){
            q.push(tree_node->left_treenode_);
            q.push(tree_node->right_treenode_);
        }
    }


    for (size_t i = 0; i < nodes_.size(); ++i) {
        delete nodes_[i];
    }
    nodes_.clear();
    data_.clear();
}

TreeNode* Hnsw::createCompleteTree(int depth, std::vector<int>& sub_node){
    // std::cout<<"depth = "<<depth<<endl;
    // std::cout<<"sub_node.size() = "<<sub_node.size()<<endl;
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    float PORTABLE_ALIGN32 TmpRes[8];

    TreeNode* newNode = new TreeNode();
    if(depth < Depth_){

        std::vector<float> data;
        data.resize(data_dim_);
        for(int i=0; i < data_dim_; i++){
            float d;
            for(int j=0; j < sub_node.size(); j++){
                d += nodes_[sub_node[j]]->GetData()[i];
            }
            data[i] = d/sub_node.size();
        }
        priority_queue<CloserFirstNew> re;
        for(int i=0; i < sub_node.size(); i++){
            float d = dist_cls_->Evaluate((float*)&(data[0]), (float*)&(nodes_[sub_node[i]]->GetData()[0]), data_dim_, TmpRes);
            re.emplace(sub_node[i], d);
        }
        int navigation_id = re.top().GetId();

        // int navigation_id = sub_node[rand() % sub_node.size()];
        priority_queue<FurtherFirstNew> res;
        for(int i=0; i < sub_node.size(); i++){
            float d = dist_cls_->Evaluate((float*)&(nodes_[sub_node[i]]->GetData()[0]), (float*)&(nodes_[navigation_id]->GetData()[0]), data_dim_, TmpRes);
            res.emplace(sub_node[i], d);
        }
        std::vector<int> right_nodes;
        right_nodes.resize(sub_node.size()/2);
        std::vector<int> left_nodes;
        left_nodes.resize(sub_node.size()-sub_node.size()/2);
        for(int i = 0; i < sub_node.size()/2; i++){
            right_nodes[i] = res.top().GetId();
            // if(depth == 2){
            //     std::cout<<res.top().GetDistance()<<"  ";
            // }
            res.pop();
        }
        std::cout<<endl;
        std::cout<<endl;
        float radius = res.top().GetDistance();
        for(int i = 0; i<sub_node.size()-sub_node.size()/2; i++){
            left_nodes[i] = res.top().GetId();
            // if(depth == 2){
            //     std::cout<<res.top().GetDistance()<<"  ";
            // }
            res.pop();
        }

        newNode->SetSubNodes(sub_node);
        newNode->SetTreeSize(sub_node.size());
        newNode->SetNavigationId(navigation_id);
        newNode->SetEnterPointId(-1);
        newNode->SetRadius(radius);
        depth += 1;
        newNode->left_treenode_ = createCompleteTree(depth, left_nodes);
        newNode->right_treenode_ = createCompleteTree(depth, right_nodes);
        return newNode;
    }else if(depth == Depth_){
        newNode->SetSubNodes(sub_node);
        newNode->SetTreeSize(sub_node.size());
        newNode->SetNavigationId(-1);
        newNode->SetRadius(0);
        std::vector<float> data;
        data.resize(data_dim_);
        for(int i=0; i < data_dim_; i++){
            float d;
            for(int j=0; j < sub_node.size(); j++){
                d += nodes_[sub_node[j]]->GetData()[i];
            }
            data[i] = d/sub_node.size();
        }
        priority_queue<CloserFirstNew> res;
        for(int i=0; i < sub_node.size(); i++){
            float d = dist_cls_->Evaluate((float*)&(data[0]), (float*)&(nodes_[sub_node[i]]->GetData()[0]), data_dim_, TmpRes);
            res.emplace(sub_node[i], d);
        }
        newNode->SetEnterPointId(res.top().GetId());
        return newNode;
    }
}

void Hnsw::BuildGraph(bool reverse) {
    nodes_.resize(data_.size());
    HnswNode* first = new HnswNode(0, &(data_[0]), MaxM_, ML_);
    nodes_[0] = first;
    enterpoint_ = first;
    if (reverse) {
        #pragma omp parallel num_threads(num_threads_)
        {
            visited_list_ = new VisitedList(data_.size());

            #pragma omp for schedule(dynamic,128)
            for (size_t i = data_.size() - 1; i >= 1; --i) {
                HnswNode* qnode = new HnswNode(i, &data_[i], MaxM_, ML_);
                nodes_[i] = qnode;
                Insert(qnode);
            }
            delete visited_list_;
            visited_list_ = nullptr;
        }
    } else {
        #pragma omp parallel num_threads(num_threads_)
        {
            visited_list_ = new VisitedList(data_.size());
            #pragma omp for schedule(dynamic,128)
            for (size_t i = 1; i < data_.size(); ++i) {
                HnswNode* qnode = new HnswNode(i, &data_[i], MaxM_, ML_);
                nodes_[i] = qnode;
                Insert(qnode);
            }
            delete visited_list_;
            visited_list_ = nullptr;
        }
    }

    search_list_.reset(new VisitedList(data_.size()));
}

bool Hnsw::SaveModel(const string& fname) const {
    ofstream b_stream(fname.c_str(), fstream::out|fstream::binary);
    if (b_stream) {
        b_stream.write(model_, model_byte_size_);
        return (b_stream.good());
    } else {
        throw std::runtime_error("[Error] Failed to save model to file: " + fname);
    }
    return false;
}

bool Hnsw::LoadModel(const string& fname, const bool use_mmap) {
    if(!use_mmap) {
        ifstream in;
        in.open(fname, fstream::in|fstream::binary|fstream::ate);
        if(in.is_open()) {
            size_t size = in.tellg();
            in.seekg(0, fstream::beg);
            model_ = new char[size];
            model_byte_size_ = size;
            in.read(model_, size);
            in.close();
        } else {
            throw std::runtime_error("[Error] Failed to load model to file: " + fname+ " not found!");
        }
    } else {
        model_mmap_ = new Mmap(fname.c_str());
        model_byte_size_ = model_mmap_->GetFileSize();
        model_ = model_mmap_->GetData();
    }
    char* ptr = model_;
    ptr = GetValueAndIncPtr<size_t>(ptr, M_);
    ptr = GetValueAndIncPtr<size_t>(ptr, MaxM_);
    ptr = GetValueAndIncPtr<size_t>(ptr, Depth_);
    ptr = GetValueAndIncPtr<size_t>(ptr, efConstruction_);
    ptr = GetValueAndIncPtr<float>(ptr, levelmult_);
    ptr = GetValueAndIncPtr<int>(ptr, enterpoint_id_);
    ptr = GetValueAndIncPtr<int>(ptr, num_nodes_);
    ptr = GetValueAndIncPtr<DistanceKind>(ptr, metric_);
    size_t model_data_dim = *((size_t*)(ptr));
    if (data_dim_ > 0 && model_data_dim != data_dim_) {
        throw std::runtime_error("[Error] index dimension(" + to_string(data_dim_)
                                 + ") != model dimension(" + to_string(model_data_dim) + ")");
    }
    ptr = GetValueAndIncPtr<size_t>(ptr, data_dim_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_data_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_link_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_tree_node_);
    long long all_node_size = memory_per_node_ * num_nodes_;
    long long model_config_size = GetModelConfigSize();
    model_node0_ = model_ + model_config_size;
    model_tree_node0_ = model_node0_ + all_node_size;
    search_list_.reset(new VisitedList(num_nodes_));
    if(dist_cls_) {
        delete dist_cls_;
    }
    switch (metric_) {
        case DistanceKind::ANGULAR:
            dist_cls_ = new AngularDistance();
            break;
        case DistanceKind::L2:
            dist_cls_ = new L2Distance();
            break;
        default:
            throw std::runtime_error("[Error] Unknown distance metric. ");
    }
    return true;
}

void Hnsw::UnloadModel() {
    if (model_mmap_ != nullptr) {
        model_mmap_->UnMap();
        delete model_mmap_;
        model_mmap_ = nullptr;
        model_ = nullptr;
        model_node0_ = nullptr;
    }

    search_list_.reset(nullptr);

    if (visited_list_ != nullptr) {
        delete visited_list_;
        visited_list_ = nullptr;
    }
}

void Hnsw::AddData(const std::vector<float>& data) {
    if (model_ != nullptr) {
        throw std::runtime_error("[Error] This index already has a trained model. Adding an item is not allowed.");
    }

    if (data.size() != data_dim_) {
        throw std::runtime_error("[Error] Invalid dimension data inserted: " + to_string(data.size()) + ", Predefined dimension: " + to_string(data_dim_));
    }

    if(metric_ == DistanceKind::ANGULAR) {
        vector<float> data_copy(data);
        NormalizeVector(data_copy);
        data_.emplace_back(data_copy);
    } else {
        data_.emplace_back(data);
    }
}

void Hnsw::Insert(HnswNode* qnode) {
    unique_lock<mutex> *lock = nullptr;
    // if (cur_level > maxlevel_) lock = new unique_lock<mutex>(max_level_guard_);

    HnswNode* enterpoint = enterpoint_;
    const std::vector<float>& qvec = qnode->GetData();
    float PORTABLE_ALIGN32 TmpRes[8];
    _mm_prefetch(&selecting_policy_cls_, _MM_HINT_T0);
    priority_queue<FurtherFirst> temp_res;

    SearchAtLayer(qvec, enterpoint, efConstruction_, temp_res);
    selecting_policy_cls_->Select(M_, temp_res, data_dim_, dist_cls_);
    while (temp_res.size() > 0) {
        auto* top_node = temp_res.top().GetNode();
        temp_res.pop();
        Link(top_node, qnode, is_naive_, data_dim_);
        Link(qnode, top_node, is_naive_, data_dim_);
    }
    if (lock != nullptr) delete lock;
}

void Hnsw::Link(HnswNode* source, HnswNode* target, bool is_naive, size_t dim) {
    std::unique_lock<std::mutex> lock(source->access_guard_);
    std::vector<HnswNode*>& neighbors = source->friends;
    std::vector<HnswNode*>& distant_neighbors = source->distant_friends;
    neighbors.push_back(target);
    bool shrink =  neighbors.size() > source->maxsize_;
    if (!shrink) return;
    float PORTABLE_ALIGN32 TmpRes[8];
    if (is_naive) {
        float max = dist_cls_->Evaluate((float*)&source->GetData()[0], (float*)&neighbors[0]->GetData()[0], dim, TmpRes);
        int maxi = 0;
        for (size_t i = 1; i < neighbors.size(); ++i) {
                float curd = dist_cls_->Evaluate((float*)&source->GetData()[0], (float*)&neighbors[i]->GetData()[0], dim, TmpRes);
                if (curd > max) {
                    max = curd;
                    maxi = i;
                }
        }
        neighbors.erase(neighbors.begin() + maxi);
    } else {
        // std::priority_queue<FurtherFirst> tempres;
        // for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
        //     _mm_prefetch((char*)&((*iter)->GetData()), _MM_HINT_T0);
        // }

        // for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
        //     tempres.emplace((*iter), dist_cls_->Evaluate((float*)&source->data_->GetData()[0], (float*)&(*iter)->GetData()[0], dim, TmpRes));
        // }
        // selecting_policy_cls_->Select(tempres.size() - 1, tempres, dim, dist_cls_);
        // neighbors.clear();
        // while (tempres.size()) {
        //     neighbors.emplace_back(tempres.top().GetNode());
        //     tempres.pop();
        // }

        std::priority_queue<FurtherFirst> tempres;
        for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
            _mm_prefetch((char*)&((*iter)->GetData()), _MM_HINT_T0);
        }

        for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
            tempres.emplace((*iter), dist_cls_->Evaluate((float*)&source->data_->GetData()[0], (float*)&(*iter)->GetData()[0], dim, TmpRes));
        }
        HnswNode* targetpoint = tempres.top().GetNode();
        selecting_policy_cls_->Select(tempres.size() - 1, tempres, dim, dist_cls_);
        neighbors.clear();
        while (tempres.size()) {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
        if(distant_neighbors.size() < ML_){
            if(IsKeep(source, targetpoint, Jump_)){
                distant_neighbors.emplace_back(tempres.top().GetNode());
            }
        }
   }
}

bool Hnsw::IsKeep(HnswNode* enterpoint, HnswNode* tagetpoint, size_t th){
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    float PORTABLE_ALIGN32 TmpRes[8];
    int jump = 0;
    HnswNode* curpoint = enterpoint;
    visited_list_->Reset();
    unsigned int mark = visited_list_->GetVisitMark();
    unsigned int* visited = visited_list_->GetVisited();
    visited[curpoint->GetId()] = mark;
    float cur_d = dist_cls_->Evaluate((float*)&tagetpoint->GetData()[0], (float*)&curpoint->GetData()[0], data_dim_, TmpRes);
    bool flag = true;
    while(flag){
        const vector<HnswNode*>& neighbors = curpoint->GetFriends();
        flag = false;
        for (size_t j = 0; j < neighbors.size(); ++j) {
            int fid = neighbors[j]->GetId();
            if (visited[fid] != mark) {
                _mm_prefetch((char*)&(neighbors[j]->GetData()), _MM_HINT_T0);
                visited[fid] = mark;
                float d = dist_cls_->Evaluate((float*)&tagetpoint->GetData()[0], (float*)&neighbors[j]->GetData()[0], data_dim_, TmpRes);
                if (d < cur_d) {
                    flag = true;
                    cur_d = d;
                    jump++;
                    curpoint = neighbors[j];
                }
            }
        }
    }
    if(jump >= th)
        return true;
    return false;
}



void Hnsw::MergeEdgesOfTwoGraphs(const vector<HnswNode*>& another_nodes) {
#pragma omp parallel for schedule(dynamic,128) num_threads(num_threads_)
    for (size_t i = 1; i < data_.size(); ++i) {
        const vector<HnswNode*>& neighbors1 = nodes_[i]->GetFriends();
        const vector<HnswNode*>& neighbors2 = another_nodes[i]->GetFriends();
        unordered_set<int> merged_neighbor_id_set = unordered_set<int>();
        for (HnswNode* cur : neighbors1) {
            merged_neighbor_id_set.insert(cur->GetId());
        }
        for (HnswNode* cur : neighbors2) {
            merged_neighbor_id_set.insert(cur->GetId());
        }
        priority_queue<FurtherFirst> temp_res;
        const std::vector<float>& ivec = data_[i].GetData();
        float PORTABLE_ALIGN32 TmpRes[8];
        for (int cur : merged_neighbor_id_set) {
            temp_res.emplace(nodes_[cur], dist_cls_->Evaluate((float*)&data_[cur].GetData()[0], (float*)&ivec[0], data_dim_, TmpRes));
        }

        // Post Heuristic
        post_policy_cls_->Select(MaxM_, temp_res, data_dim_, dist_cls_);
        vector<HnswNode*> merged_neighbors = vector<HnswNode*>();
        while (!temp_res.empty()) {
            merged_neighbors.emplace_back(temp_res.top().GetNode());
            temp_res.pop();
        }
        nodes_[i]->SetFriends(merged_neighbors);
    }
}

void Hnsw::NormalizeVector(std::vector<float>& vec) {
   float sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
   if (sum != 0.0) {
       sum = 1 / sqrt(sum);
       std::transform(vec.begin(), vec.end(), vec.begin(), std::bind1st(std::multiplies<float>(), sum));
   }
}

bool Hnsw::SetValuesFromModel(char* model) {
    if(model) {
        char* ptr = model;
        ptr = GetValueAndIncPtr<size_t>(ptr, M_);
        ptr = GetValueAndIncPtr<size_t>(ptr, MaxM_);
        ptr = GetValueAndIncPtr<size_t>(ptr, efConstruction_);
        ptr = GetValueAndIncPtr<float>(ptr, levelmult_);
        ptr = GetValueAndIncPtr<int>(ptr, enterpoint_id_);
        ptr = GetValueAndIncPtr<int>(ptr, num_nodes_);
        ptr = GetValueAndIncPtr<DistanceKind>(ptr, metric_);
        ptr += sizeof(size_t);
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_data_);
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_link_);
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_);
        long long all_node_size = memory_per_node_ * num_nodes_;
        long long model_config_size = GetModelConfigSize();
        model_node0_ = model_ + model_config_size;
        return true;
    }
    return false;
}
void Hnsw::SearchByVector(const vector<float>& qvec, size_t k, size_t ef_search, size_t alg, vector<pair<int, float>>& result) {
    if (model_ == nullptr) throw std::runtime_error("[Error] Model has not loaded!");
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* qraw = nullptr;

    if (ef_search < 0) {
        ef_search = 50 * k;
    }

    unsigned int mark = search_list_->GetVisitMark();
    unsigned int* visited = search_list_->GetVisited();

    priority_queue<FurtherFirstNew> res ;
    priority_queue<CloserFirstNew> candidates;

    vector<float> qvec_copy(qvec);
    if(metric_ == DistanceKind::ANGULAR) {
        NormalizeVector(qvec_copy);
    }

    qraw = &qvec_copy[0];
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    

    float d;
    bool changed;
    int id = *((int*)model_tree_node0_);
    int nav_id = *((int*)(model_tree_node0_ + sizeof(int)));
    float radius = *((float*)(model_tree_node0_ + 2*sizeof(int)));
    int enter_id = *((int*)(model_tree_node0_ + 2*sizeof(int) + sizeof(float)));
    while(enter_id == -1){
        d = (dist_cls_->Evaluate(qraw, (float *)(model_node0_ + nav_id * memory_per_node_ + memory_per_link_), data_dim_, TmpRes));
        comparison_times_ += 1;
        res.emplace(nav_id, d);
        candidates.emplace(nav_id, d);
        // visited[nav_id] = mark;
        // std::cout<<d<<endl;
        if(d > radius){
            // std::cout<<"走右子树"<<endl;
            id = *((int*)(model_tree_node0_ + (2*id + 1 - 1)*memory_per_tree_node_));;
            nav_id = *((int*)(model_tree_node0_ + (2*id + 1 - 1)*memory_per_tree_node_ + sizeof(int)));
            radius = *((float*)(model_tree_node0_ + (2*id + 1 - 1)*memory_per_tree_node_ + 2*sizeof(int)));
            // std::cout<<radius<<endl;
            enter_id = *((int*)(model_tree_node0_ + (2*id + 1 - 1)*memory_per_tree_node_ + 2*sizeof(int) + sizeof(float)));
        }else{
            // std::cout<<"走左子树"<<endl;
            id = *((int*)(model_tree_node0_ + (2*id - 1)*memory_per_tree_node_));;
            nav_id = *((int*)(model_tree_node0_ + (2*id - 1)*memory_per_tree_node_ + sizeof(int)));
            radius = *((float*)(model_tree_node0_ + (2*id - 1)*memory_per_tree_node_ + 2*sizeof(int)));
            // std::cout<<radius<<endl;
            enter_id = *((int*)(model_tree_node0_ + (2*id - 1)*memory_per_tree_node_ + 2*sizeof(int) + sizeof(float)));
        }
    }

    int cur_node_id = enter_id;
    float cur_dist = dist_cls_->Evaluate(qraw, (float *)(model_node0_ + cur_node_id*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);

    res.emplace(cur_node_id, cur_dist);
    candidates.emplace(cur_node_id, cur_dist);

    cur_node_id = candidates.top().GetId();
    cur_dist = candidates.top().GetDistance();

    bool flag = 1;

    while(flag == 1) {
        flag = 0;
        float d;
        // const CloserFirstNew& cand = candidates.top();
        // candidates.pop();
        // float lowerbound = res.top().GetDistance();
        // if (cand.GetDistance() > lowerbound) break;
        // cur_node_id = cand.GetId();
        // cur_dist = cand.GetDistance();



        float redius = *((float*)(model_node0_ + cur_node_id*memory_per_node_));
        // std::cout<<"redius: "<<redius<<endl;
        // std::cout<<"cur_node_id: "<<cur_node_id<<endl;
        // std::cout<<"cur_dist: "<<cur_dist<<endl;
        int left_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_ + sizeof(float)));
        int *left_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float));
        int right_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) +sizeof(int) + sizeof(int)*(left_size)));
        int *right_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) + sizeof(int) + sizeof(int)*(left_size));

        if(cur_dist - redius > 0 ){
            int tnum;
            for (int j = 1; j <= right_size; ++j) {
                tnum = *(right_address + j);
                _mm_prefetch(dist_cls_, _MM_HINT_T0);
                if (visited[tnum] != mark) {
                    visited[tnum] = mark;
                    d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
                    comparison_times_ += 1;
                    if(res.size() < ef_search || d < res.top().GetDistance()){
                        res.emplace(tnum, d);
                        if (res.size() > ef_search) res.pop();
                    }
                    if(candidates.size() < ef_search || d < res.top().GetDistance()){
                        candidates.emplace(tnum, d);
                    }
                    if(cur_dist > d){
                        cur_node_id = tnum;
                        cur_dist = d;
                        flag = true;
                    }
                }
            }
            if(flag == true){
                // std::cout<<"情况一:一半"<<endl;
            }else{
                // std::cout<<"情况一:全部"<<endl;
            }
            if(flag == false){
                int tnum;
                for (int j = 1; j <= left_size; ++j) {
                    tnum = *(left_address + j);
                    _mm_prefetch(dist_cls_, _MM_HINT_T0);
                    if (visited[tnum] != mark) {
                        visited[tnum] = mark;
                        d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
                        comparison_times_ += 1;
                        if(res.size() < ef_search || d < res.top().GetDistance()){
                            res.emplace(tnum, d);
                            if (res.size() > ef_search) res.pop();
                        }
                        if(candidates.size() < ef_search || d < res.top().GetDistance()){
                            candidates.emplace(tnum, d);
                        }
                        if(cur_dist > d){
                            cur_node_id = tnum;
                            cur_dist = d;
                            flag = true;
                        }
                    }
                }
            }
        }else{

            int tnum;
            for (int j = 1; j <= left_size; ++j) {
                tnum = *(left_address + j);
                _mm_prefetch(dist_cls_, _MM_HINT_T0);
                if (visited[tnum] != mark) {
                    visited[tnum] = mark;
                    d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
                    comparison_times_ += 1;
                    if(res.size() < ef_search || d < res.top().GetDistance()){
                        res.emplace(tnum, d);
                        if (res.size() > ef_search) res.pop();
                    }
                    if(candidates.size() < ef_search || d < res.top().GetDistance()){
                        candidates.emplace(tnum, d);
                    }
                    if(cur_dist > d){
                        cur_node_id = tnum;
                        cur_dist = d;
                        flag = true;
                    }
                }
            }
            if(flag == true){
                // std::cout<<"情况2:一半"<<endl;
            }else{
                // std::cout<<"情况2:全部"<<endl;
            }
            if(flag == false){
                int tnum;
                for (int j = 1; j <= right_size; ++j) {
                    tnum = *(right_address + j);
                    _mm_prefetch(dist_cls_, _MM_HINT_T0);
                    if (visited[tnum] != mark) {
                        visited[tnum] = mark;
                        d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
                        comparison_times_ += 1;
                        if(res.size() < ef_search || d < res.top().GetDistance()){
                            res.emplace(tnum, d);
                            if (res.size() > ef_search) res.pop();
                        }
                        if(candidates.size() < ef_search || d < res.top().GetDistance()){
                            candidates.emplace(tnum, d);
                        }
                        if(cur_dist > d){
                            cur_node_id = tnum;
                            cur_dist = d;
                            flag = true;
                        }
                    }
                }
            }
            // std::cout<<"情况2:全部"<<endl;
            // int tnum;
            // for (int j = 1; j <= left_size; ++j) {
            //     tnum = *(left_address + j);
            //     _mm_prefetch(dist_cls_, _MM_HINT_T0);
            //     if (visited[tnum] != mark) {
            //         visited[tnum] = mark;
            //         d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
            //         comparison_times_ += 1;
            //         if(res.size() < ef_search || d < res.top().GetDistance()){
            //             res.emplace(tnum, d);
            //             if (res.size() > ef_search) res.pop();
            //         }
            //         if(candidates.size() < ef_search || d < res.top().GetDistance()){
            //             candidates.emplace(tnum, d);
            //         }
            //         if(cur_dist > d){
            //             cur_node_id = tnum;
            //             cur_dist = d;
            //             flag = true;
            //         }
            //     }
            // }

            // for (int j = 1; j <= right_size; ++j) {
            //     tnum = *(right_address + j);
            //     _mm_prefetch(dist_cls_, _MM_HINT_T0);
            //     if (visited[tnum] != mark) {
            //         visited[tnum] = mark;
            //         d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
            //         comparison_times_ += 1;
            //         if(res.size() < ef_search || d < res.top().GetDistance()){
            //             res.emplace(tnum, d);
            //             if (res.size() > ef_search) res.pop();
            //         }
            //         if(candidates.size() < ef_search || d < res.top().GetDistance()){
            //             candidates.emplace(tnum, d);
            //         }
            //         if(cur_dist > d){
            //             cur_node_id = tnum;
            //             cur_dist = d;
            //             flag = true;
            //         }
            //     }
            // }
        }
    }


    vector<pair<int, float> > path;
    if (ensure_k_) path.emplace_back(cur_node_id, cur_dist);
    // for (int i = maxlevel; i > 0; --i) {
    //     changed = true;
    //     while (changed) {
    //         changed = false;
    //         char* node_offset = model_node0_ + cur_node_id*memory_per_node_;
    //         int *data = (int*)(level_base_offset + (i-1) * memory_per_node_higher_level_);
    //         int size = *data;

    //         for (int j = 1; j <= size; ++j) {
    //             int tnum = *(data + j);
    //             d = (dist_cls_->Evaluate(qraw, (float *)(model_level0_ + tnum*memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes));
    //             if (d < cur_dist) {
    //                 cur_dist = d;
    //                 cur_node_id = tnum;
    //                 offset = *((int*)(model_level0_ + cur_node_id*memory_per_node_level0_));
    //                 changed = true;
    //                 if (ensure_k_) path.emplace_back(cur_node_id, cur_dist);
    //              }
    //         }
    //     }
    // }

    if (ensure_k_) {
        while (result.size() < k && !path.empty()) {
            cur_node_id = path.back().first;
            cur_dist = path.back().second;
            path.pop_back();
            if(alg == 1){
                SearchById_(cur_node_id, cur_dist, qraw, k, ef_search, result);
            }else{
                SearchById_true(cur_node_id, cur_dist, qraw, visited, res, candidates, k, ef_search, result);
            }
        }
    } else {
        if(alg == 1){
                SearchById_(cur_node_id, cur_dist, qraw, k, ef_search, result);
            }else{
                SearchById_true(cur_node_id, cur_dist, qraw, visited, res, candidates, k, ef_search, result);
            }
    }
}

void Hnsw::SearchById_true(int cur_node_id, float cur_dist, const float* qraw, unsigned int* visited, priority_queue<FurtherFirstNew>& res, priority_queue<CloserFirstNew>& candidates, size_t k, size_t ef_search, vector<pair<int, float> >& result) {
    float PORTABLE_ALIGN32 TmpRes[8];
    unsigned int mark = search_list_->GetVisitMark();
    bool need_sort = false;
    bool flag = cur_node_id == candidates.top().GetId();
    // std::cout<<"flag = "<<flag<<endl;
    if (ensure_k_) {
        if (!result.empty()) need_sort = true;
        for (size_t i = 0; i < result.size(); ++i)
            visited[result[i].first] = mark;
        if (visited[cur_node_id] == mark) return;
    }
    visited[cur_node_id] = mark;
    while(!candidates.empty()) {
        const CloserFirstNew& cand = candidates.top();
        float lowerbound = res.top().GetDistance();
        if (cand.GetDistance() > lowerbound) break;
        int cur_node_id = cand.GetId();
        float cur_dist = cand.GetDistance();

        float redius = *((float*)(model_node0_ + cur_node_id*memory_per_node_));
        int left_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_ + sizeof(float)));
        int *left_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float));
        int right_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) +sizeof(int) + sizeof(int)*(left_size)));
        int *right_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) + sizeof(int) + sizeof(int)*(left_size));

        candidates.pop();

        int tnum;
        float d;
        for (int j = 1; j <= left_size; ++j) {
            tnum = *(left_address + j);
            _mm_prefetch(dist_cls_, _MM_HINT_T0);
            if (visited[tnum] != mark) {
                // std::cout<<"执行到了!"<<flag<<endl;
                visited[tnum] = mark;
                d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
                comparison_times_ += 1;
                if (res.size() < ef_search || res.top().GetDistance() > d) {
                    res.emplace(tnum, d);
                    candidates.emplace(tnum, d);
                    if (res.size() > ef_search) res.pop();
                }
            }
        }

        for (int j = 1; j <= right_size; ++j) {
            tnum = *(right_address + j);
            _mm_prefetch(dist_cls_, _MM_HINT_T0);
            if (visited[tnum] != mark) {
                visited[tnum] = mark;
                d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
                comparison_times_ += 1;
                if (res.size() < ef_search || res.top().GetDistance() > d) {
                    res.emplace(tnum, d);
                    candidates.emplace(tnum, d);
                    if (res.size() > ef_search) res.pop();
                }
            }
        }
    }

    // while(!candidates.empty()) {
    //     float d;
    //     const CloserFirstNew& cand = candidates.top();
    //     candidates.pop();
    //     float lowerbound = res.top().GetDistance();
    //     if (cand.GetDistance() > lowerbound) break;
    //     cur_node_id = cand.GetId();
    //     cur_dist = cand.GetDistance();


    //     float redius = *((float*)(model_node0_ + cur_node_id*memory_per_node_));
    //     // std::cout<<"redius: "<<redius<<endl;
    //     // std::cout<<"cur_node_id: "<<cur_node_id<<endl;
    //     // std::cout<<"cur_dist: "<<cur_dist<<endl;
    //     int left_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_ + sizeof(float)));
    //     int *left_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float));
    //     int right_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) +sizeof(int) + sizeof(int)*(left_size)));
    //     int *right_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) + sizeof(int) + sizeof(int)*(left_size));

    //     if(cur_dist - redius > 100 ){
    //         // std::cout<<"情况1"<<endl;
    //         int tnum;
    //         priority_queue<CloserFirstNew> little_candidates;
    //         for (int j = 1; j <= right_size; ++j) {
    //             tnum = *(right_address + j);
    //             _mm_prefetch(dist_cls_, _MM_HINT_T0);
    //             if (visited[tnum] != mark) {
    //                 visited[tnum] = mark;
    //                 // std::cout<<tnum<<endl;
    //                 d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
    //                 comparison_times_ += 1;
    //                 little_candidates.emplace(tnum, d);
    //             }
    //         }
    //         if(little_candidates.size()==0){
    //             continue;
    //         }
    //         if(little_candidates.top().GetDistance() < cur_dist){
    //             harf += 1;
    //             // std::cout<<"情况1:一半"<<endl;
    //             while(!little_candidates.empty()){
    //                 if (res.size() < ef_search || res.top().GetDistance() > little_candidates.top().GetDistance()) {
    //                     res.emplace(little_candidates.top().GetId(), little_candidates.top().GetDistance());
    //                     candidates.emplace(little_candidates.top().GetId(), little_candidates.top().GetDistance());
    //                     if (res.size() > ef_search) res.pop();
    //                 }
    //                 little_candidates.pop();
    //             }
    //         }else{
    //             // std::cout<<"情况1:全部"<<endl;
    //             int tnum;
    //             priority_queue<CloserFirstNew> little_candidates;
    //             for (int j = 1; j <= left_size; ++j) {
    //                 std::cout<<j<<endl;
    //                 tnum = *(left_address + j);
    //                 _mm_prefetch(dist_cls_, _MM_HINT_T0);
    //                 if (visited[tnum] != mark) {
    //                     visited[tnum] = mark;
    //                     d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
    //                     comparison_times_ += 1;
    //                     little_candidates.emplace(tnum, d);
    //                 }
    //             }
    //             if(little_candidates.top().GetDistance() < cur_dist){
    //                 while(!little_candidates.empty()){
    //                     if (res.size() < ef_search || res.top().GetDistance() > d) {
    //                         res.emplace(little_candidates.top().GetId(), little_candidates.top().GetDistance());
    //                         candidates.emplace(little_candidates.top().GetId(), little_candidates.top().GetDistance());
    //                         if (res.size() > ef_search) res.pop();
    //                     }
    //                     little_candidates.pop();
    //                 }
    //             }
    //         }
    //         std::cout<<endl;
    //     }else{
    //         // std::cout<<"情况2:全部"<<endl;
    //         int tnum;
    //         for (int j = 1; j <= left_size; ++j) {
    //             tnum = *(left_address + j);
    //             _mm_prefetch(dist_cls_, _MM_HINT_T0);
    //             if (visited[tnum] != mark) {
    //                 visited[tnum] = mark;
    //                 d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
    //                 comparison_times_ += 1;
    //                 if (res.size() < ef_search || res.top().GetDistance() > d) {
    //                     res.emplace(tnum, d);
    //                     candidates.emplace(tnum, d);
    //                     if (res.size() > ef_search) res.pop();
    //                 }
    //             }
    //         }

    //         for (int j = 1; j <= right_size; ++j) {
    //             tnum = *(right_address + j);
    //             _mm_prefetch(dist_cls_, _MM_HINT_T0);
    //             if (visited[tnum] != mark) {
    //                 visited[tnum] = mark;
    //                 d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
    //                 comparison_times_ += 1;
    //                 if (res.size() < ef_search || res.top().GetDistance() > d) {
    //                     res.emplace(tnum, d);
    //                     candidates.emplace(tnum, d);
    //                     if (res.size() > ef_search) res.pop();
    //                 }
    //             }
    //         }
    //         std::cout<<endl;
    //     }
    // }

    float r = 100;

    // while(!candidates.empty()){
    //     const CloserFirstNew& cand = candidates.top();
    //     candidates.pop();
    //     float lowerbound = res.top().GetDistance();
    //     cur_node_id = cand.GetId();
    //     cur_dist = cand.GetDistance();
    //     if (cur_dist > 1.01*r) break;


    //     float redius = *((float*)(model_node0_ + cur_node_id*memory_per_node_));
    //     // std::cout<<"redius: "<<redius<<endl;
    //     // std::cout<<"cur_node_id: "<<cur_node_id<<endl;
    //     // std::cout<<"cur_dist: "<<cur_dist<<endl;
    //     int left_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_ + sizeof(float)));
    //     int *left_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float));
    //     int right_size = *((int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) +sizeof(int) + sizeof(int)*(left_size)));
    //     int *right_address = (int*)(model_node0_ + cur_node_id*memory_per_node_+ sizeof(float) + sizeof(int) + sizeof(int)*(left_size));

    //     int tnum;
    //     float d;
    //     for (int j = 1; j <= left_size; ++j) {
    //         tnum = *(left_address + j);
    //         _mm_prefetch(dist_cls_, _MM_HINT_T0);
    //         if (visited[tnum] != mark) {
    //             visited[tnum] = mark;
    //             d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
    //             comparison_times_ += 1;
    //             if (d < 1.01*r) {
    //                 candidates.emplace(tnum, d);
    //             }
    //             if(d < r){
    //                 res.emplace(tnum, d);
    //                 if(res.size() > ef_search){
    //                     res.pop();
    //                 }
    //                 if(res.size() == ef_search){
    //                     r = res.top().GetDistance();
    //                 }
    //             }
    //         }
    //     }

    //     for (int j = 1; j <= right_size; ++j) {
    //         tnum = *(right_address + j);
    //         _mm_prefetch(dist_cls_, _MM_HINT_T0);
    //         if (visited[tnum] != mark) {
    //             visited[tnum] = mark;
    //             d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
    //             comparison_times_ += 1;
    //             if (d < 1.01*r) {
    //                 candidates.emplace(tnum, d);
    //             }
    //             if(d < r){
    //                 res.emplace(tnum, d);
    //                 if(res.size() > ef_search){
    //                     res.pop();
    //                 }
    //                 if(res.size() == ef_search){
    //                     r = res.top().GetDistance();
    //                 }
    //             }
    //         }
    //     }   
    // }






    while(res.size()>k){
        res.pop();
    }

    while(!res.empty()){
        FurtherFirstNew ff = res.top();
        result.push_back(pair<int, float>(ff.GetId(), ff.GetDistance()));
        res.pop();
    }


    sort(result.begin(), result.end(), [](const pair<int, float>& i, const pair<int, float>& j) -> bool {
                return i.second < j.second; });

}

void Hnsw::SearchById_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search, vector<pair<int, float> >& result) {
    MinHeap<float, int> dh;
    dh.push(cur_dist, cur_node_id);
    float PORTABLE_ALIGN32 TmpRes[8];

    typedef typename MinHeap<float, int>::Item  QueueItem;
    std::queue<QueueItem> q;
    search_list_->Reset();


    unsigned int mark = search_list_->GetVisitMark();
    unsigned int* visited = search_list_->GetVisited();
    bool need_sort = false;
    if (ensure_k_) {
        if (!result.empty()) need_sort = true;
        for (size_t i = 0; i < result.size(); ++i)
            visited[result[i].first] = mark;
        if (visited[cur_node_id] == mark) return;
    }
    visited[cur_node_id] = mark;

    std::priority_queue<pair<float, int> > visited_nodes;

    int tnum;
    float d;
    QueueItem e;
    float maxKey = cur_dist;
    size_t total_size = 1;
    while (dh.size() > 0 && visited_nodes.size() < (ef_search >> 1)) {
        e = dh.top();
        dh.pop();
        cur_node_id = e.data;

        visited_nodes.emplace(e.key, e.data);   

        float topKey = maxKey;

        int *data = (int*)(model_node0_ + cur_node_id*memory_per_node_);
        int size = *data;
        for (int j = 1; j <= size; ++j) {
            tnum = *(data + j);
            _mm_prefetch(dist_cls_, _MM_HINT_T0);
            if (visited[tnum] != mark) {
                visited[tnum] = mark;
                d = dist_cls_->Evaluate(qraw, (float*)(model_node0_ + tnum*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
                comparison_times_ += 1;
                if (d < topKey || total_size < ef_search) {
                    // topKey = d;
                    q.emplace(QueueItem(d, tnum));
                    ++total_size;
                }
            }
        }
        while(!q.empty()) {
            dh.push(q.front().key, q.front().data);
            if (q.front().key > maxKey) maxKey = q.front().key;
            q.pop();
        }
    }
    // std::cout<<"visited_nodes.size() = "<<visited_nodes.size()<<endl;

    // std::cout<<"ef_search："<<ef_search<<endl;

    // std::cout<<"total_size："<<total_size<<endl;

    // std::cout<<"比较次数："<<comparison_times_<<endl;

    vector<pair<float, int> > res_t;
    while(dh.size() && res_t.size() < k) {
        res_t.emplace_back(dh.top().key, dh.top().data);
        dh.pop();
    }
    while (visited_nodes.size() > k) visited_nodes.pop();
    while (!visited_nodes.empty()) {
        res_t.emplace_back(visited_nodes.top());
        visited_nodes.pop();
    }
    _mm_prefetch(&res_t[0], _MM_HINT_T0);
    std::sort(res_t.begin(), res_t.end());
    size_t sz;
    if (ensure_k_) {
        sz = min(k - result.size(), res_t.size());
    } else {
        sz = min(k, res_t.size());
    }
    for(size_t i = 0; i < sz; ++i)
        result.push_back(pair<int, float>(res_t[i].second, res_t[i].first));
    if (ensure_k_ && need_sort) {
        _mm_prefetch(&result[0], _MM_HINT_T0);
        sort(result.begin(), result.end(), [](const pair<int, float>& i, const pair<int, float>& j) -> bool {
                return i.second < j.second; });
    }
}

void Hnsw::SearchById(int id, size_t k, size_t ef_search, vector<pair<int, float> >& result) {
    if (ef_search < 0) {
        ef_search = 50 * k;
    }
    SearchById_(id, 0.0, (const float*)(model_node0_ + id * memory_per_node_+ memory_per_link_), k, ef_search, result);
}

void Hnsw:: SearchByVector_violence(const std::vector<float>& qvec, size_t k, size_t ef_search, std::vector<std::pair<int, float>>& result){
 if (model_ == nullptr) throw std::runtime_error("[Error] Model has not loaded!");
    // TODO: check Node 12bytes => 8bytes
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* qraw = nullptr;

    // priority_queue<CloserFirstNew> candidates;
    priority_queue<FurtherFirstNew> res;
    if (ef_search < 0) {
        ef_search = 50 * k;
    }
    vector<float> qvec_copy(qvec);
    if(metric_ == DistanceKind::ANGULAR) {
        NormalizeVector(qvec_copy);
    }
    // 查询点向量
    qraw = &qvec_copy[0];
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    // int maxlevel = maxlevel_;
    // 当前离查询最近的点的id
    int cur_node_id = enterpoint_id_;
    // 当前离查询最近点的距离
    float cur_dist;
    
    float d;

    search_list_->Reset();

    vector<pair<int, float> > path;
    if (ensure_k_) path.emplace_back(cur_node_id, cur_dist);

    for(int i=0;i<num_nodes_;i++){
        char* current_node_address = model_node0_ + i*memory_per_node_;
        cur_dist = dist_cls_->Evaluate(qraw, (float *)(model_node0_ + i*memory_per_node_ + memory_per_link_), data_dim_, TmpRes);
        if(res.size() < k || res.top().GetDistance() > cur_dist){
            res.emplace(i,cur_dist);
            if(res.size() > k) res.pop();
        }
    }

    // std::cout<<res.size()<<endl;

    while(!res.empty()){
        const FurtherFirstNew& temp = res.top();
        result.push_back(pair<int, float>(temp.GetId(), temp.GetDistance()));
        res.pop();
    }
}


void Hnsw::SearchAtLayer(const std::vector<float>& qvec, HnswNode* enterpoint, size_t ef, priority_queue<FurtherFirst>& result) {
    // TODO: check Node 12bytes => 8bytes
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* qraw = &qvec[0];
    priority_queue<CloserFirst> candidates;
    float d = dist_cls_->Evaluate(qraw, (float*)&(enterpoint->GetData()[0]), data_dim_, TmpRes);
    result.emplace(enterpoint, d);
    candidates.emplace(enterpoint, d);

    visited_list_->Reset();
    unsigned int mark = visited_list_->GetVisitMark();
    unsigned int* visited = visited_list_->GetVisited();
    visited[enterpoint->GetId()] = mark;

    while(!candidates.empty()) {
        const CloserFirst& cand = candidates.top();
        float lowerbound = result.top().GetDistance();
        if (cand.GetDistance() > lowerbound) break;
        HnswNode* cand_node = cand.GetNode();
        unique_lock<mutex> lock(cand_node->access_guard_);
        const vector<HnswNode*>& neighbors = cand_node->GetFriends();
        candidates.pop();
        for (size_t j = 0; j < neighbors.size(); ++j) {
            _mm_prefetch((char*)&(neighbors[j]->GetData()), _MM_HINT_T0);
        }
        for (size_t j = 0; j < neighbors.size(); ++j) {
            int fid = neighbors[j]->GetId();
            if (visited[fid] != mark) {
                _mm_prefetch((char*)&(neighbors[j]->GetData()), _MM_HINT_T0);
                visited[fid] = mark;
                d = dist_cls_->Evaluate(qraw, (float*)&neighbors[j]->GetData()[0], data_dim_, TmpRes);
                if (result.size() < ef || result.top().GetDistance() > d) {
                    result.emplace(neighbors[j], d);
                    candidates.emplace(neighbors[j], d);
                    if (result.size() > ef) result.pop();
                }
            }
        }
    }
}

size_t Hnsw::GetModelConfigSize() const {
    size_t ret = 0;
    ret += sizeof(M_);
    ret += sizeof(MaxM_);
    ret += sizeof(Depth_);
    ret += sizeof(efConstruction_);
    ret += sizeof(levelmult_);
    ret += sizeof(enterpoint_id_);
    ret += sizeof(num_nodes_);
    ret += sizeof(metric_);
    ret += sizeof(data_dim_);
    ret += sizeof(memory_per_data_);
    ret += sizeof(memory_per_link_);
    ret += sizeof(memory_per_node_);
    ret += sizeof(memory_per_tree_node_);
    return ret;
}

void Hnsw::SaveModelConfig(char* ptr) {
    ptr = SetValueAndIncPtr<size_t>(ptr, M_);
    ptr = SetValueAndIncPtr<size_t>(ptr, MaxM_);
    ptr = SetValueAndIncPtr<size_t>(ptr, Depth_);
    ptr = SetValueAndIncPtr<size_t>(ptr, efConstruction_);
    ptr = SetValueAndIncPtr<float>(ptr, levelmult_);
    ptr = SetValueAndIncPtr<int>(ptr, enterpoint_id_);
    ptr = SetValueAndIncPtr<int>(ptr, num_nodes_);
    ptr = SetValueAndIncPtr<DistanceKind>(ptr, metric_);
    ptr = SetValueAndIncPtr<size_t>(ptr, data_dim_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_data_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_link_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_node_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_tree_node_);
}

void Hnsw::PrintConfigs() const {
    logger_->info("HNSW configurations & status: M({}), MaxM({}), MaxM0({}), efCon({}), levelmult({}), maxlevel({}), #nodes({}), dimension of data({}), memory per data({}), memory per link level0({}), memory per node level0({}), memory per node higher level({}), higher level offset({}), level0 offset({})", M_, MaxM_, efConstruction_, levelmult_, num_nodes_, data_dim_, memory_per_data_, memory_per_link_, memory_per_node_);
}

void Hnsw::PrintDegreeDist() const {
    logger_->info("* Degree distribution");
    vector<int> degrees(MaxM_ + 2, 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        degrees[nodes_[i]->GetFriends().size()]++;
    }
    for (size_t i = 0; i < degrees.size(); ++i) {
        logger_->info("degree: {}, count: {}", i, degrees[i]);
    }
}

} // namespace n2
