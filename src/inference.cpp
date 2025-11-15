#include "mf_common.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>

using namespace std;

void recommend_top_k_for_user(const MFModel& model,
                              std::uint32_t  user_id,
                              int            K)
{
    if (user_id >= model.n_users) {
        cerr << "[inference] Invalid user_id: " << user_id
             << " (n_users = " << model.n_users << ")\n";
        return;
    }
    if (K <= 0) {
        cerr << "[inference] K must be positive.\n";
        return;
    }
    if (model.n_items == 0) {
        cerr << "[inference] Model has zero items.\n";
        return;
    }

    vector<pair<std::uint32_t, float>> scores;
    scores.reserve(model.n_items);

    for (std::uint32_t item = 0; item < model.n_items; ++item) {
        float pred = predict_rating(model, user_id, item);
        scores.emplace_back(item, pred);
    }

    if (K > static_cast<int>(scores.size())) {
        K = static_cast<int>(scores.size());
    }

    std::nth_element(
        scores.begin(),
        scores.begin() + K,
        scores.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );

    std::sort(
        scores.begin(),
        scores.begin() + K,
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );

    cout << "Top " << K << " recommended items for user " << user_id << ":\n";
    for (int idx = 0; idx < K; ++idx) {
        auto [item, score] = scores[idx];
        cout << "  item " << item << "  | predicted rating = " << score << "\n";
    }
}
