#ifndef MF_COMMON_H
#define MF_COMMON_H

#include <cstdint>
#include <vector>
#include <string>

struct Rating {
    std::uint32_t u;
    std::uint32_t i; 
    float         r;
};

struct Dataset {
    std::vector<Rating> train;
    std::vector<Rating> test;

    std::uint32_t n_users = 0;
    std::uint32_t n_items = 0;

    float global_mean = 0.0f; 
};

struct TrainConfig {
    int   k          = 50; 
    float eta        = 0.01f; 
    float lambda     = 0.05f; 
    int   epochs     = 20;    

    float test_ratio = 0.2f; 
};

struct MFModel {
    std::vector<float> U; 
    std::vector<float> V;  
    std::vector<float> bu;
    std::vector<float> bi; 

    std::uint32_t n_users = 0;
    std::uint32_t n_items = 0;
    int           k       = 0;

    float global_mean = 0.0f; 
};

Dataset load_movielens(const std::string& ratings_path,
                       float              test_ratio = 0.2f,
                       std::uint32_t      random_seed = 42);

MFModel train_omp(const Dataset& ds, const TrainConfig& cfg);

MFModel train_mpi(const Dataset& ds, const TrainConfig& cfg);

float predict_rating(const MFModel& model,
                     std::uint32_t  u,
                     std::uint32_t  i);

double compute_rmse(const std::vector<Rating>& data,
                    const MFModel&             model);

MFModel run_training_omp(const std::string& ratings_path,
                         const TrainConfig& cfg);

MFModel run_training_mpi(const std::string& ratings_path,
                         const TrainConfig& cfg);

void recommend_top_k_for_user(const MFModel& model,
                              std::uint32_t  user_id,
                              int            K);

#endif 