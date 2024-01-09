

#include "../../hnswlib/hnswlib.h"
#include "cnpy.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <utility>

using namespace hnswlib;

typedef size_t labeltype;

void dump_index_to_matrix_market(HierarchicalNSW<float> &appr_alg,
                                 const char *filename, int num_edges) {
  std::ofstream output_file;
  output_file.open(filename);

  std::vector<std::vector<size_t>> graph = appr_alg.getGraph();

  output_file << "\%\%MatrixMarket matrix coordinate integer general"
              << std::endl;
  output_file << appr_alg.cur_element_count << " " << appr_alg.cur_element_count
              << " " << num_edges << std::endl;

  for (int i = 0; i < appr_alg.cur_element_count; i++) {
    for (int j = 0; j < graph[i].size(); j++) {
      output_file << i + 1 << " " << graph[i][j] + 1 << std::endl;
    }
  }

  output_file.close();
}

void get_gt(
    unsigned int *ground_truth, float *queries, float *training_data,
    size_t qsize, L2Space &space, size_t vecdim,
    std::vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {

  (std::vector<std::priority_queue<std::pair<float, labeltype>>>(qsize))
      .swap(answers);

  DISTFUNC<float> fstdistfunc_ = space.get_dist_func();

  for (int i = 0; i < qsize; i++) {
    for (int j = 0; j < k; j++) {
      float other =
          fstdistfunc_(queries + i * vecdim,
                       training_data + ground_truth[100 * i + j] * vecdim,
                       space.get_dist_func_param());
      answers[i].emplace(other, ground_truth[100 * i + j]);
    }
  }
}

static float test_approx(
    float *queries, size_t qsize, HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    std::vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
  size_t correct = 0;
  size_t total = 0;

  for (int i = 0; i < qsize; i++) {
    std::priority_queue<std::pair<float, labeltype>> result =
        appr_alg.searchKnn(queries + vecdim * i, k);
    std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
    std::unordered_set<labeltype> g;
    total += gt.size();

    while (gt.size()) {
      g.insert(gt.top().second);
      gt.pop();
    }

    while (result.size()) {
      if (g.find(result.top().second) != g.end()) {
        correct++;
      } else {
      }
      result.pop();
    }
  }
  return 1.0f * correct / total;
}

static void test_vs_recall(
    float *queries, size_t qsize, HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    std::vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    std::vector<size_t> &ef_searches, size_t k) {

  for (size_t ef : ef_searches) {
    appr_alg.setEf(ef);

    float recall = test_approx(queries, qsize, appr_alg, vecdim, answers, k);
    std::cout << "[info] Recall: " << recall << " ef-search: " << ef
              << std::endl;
  }
}

void run_test(const std::string &train_data_path,
              const std::string &queries_path,
              const std::string &groundtruth_path,
              const std::string &mtx_filename, int ef_construction = 100,
              bool run_search = true) {
  int M = 16;

  cnpy::NpyArray arr = cnpy::npy_load(train_data_path.c_str());
  float *training_data = arr.data<float>();

  cnpy::NpyArray arr2 = cnpy::npy_load(queries_path.c_str());
  float *queries = arr2.data<float>();

  cnpy::NpyArray arr3 = cnpy::npy_load(groundtruth_path.c_str());
  unsigned int *groundtruth_ids = arr3.data<unsigned int>();

  uint32_t num_datapoints = arr.shape[0];
  uint32_t vecdim = arr.shape[1];

  std::cout << "num_datapoints: " << num_datapoints << std::endl;
  std::cout << "vecdim: " << vecdim << std::endl;

  std::cout << "constructing the index" << std::endl;
  std::cout << "ef_construction: " << ef_construction << std::endl;

  L2Space l2space(vecdim);
  // InnerProductSpace ip_space(vecdim);
  HierarchicalNSW<float> appr_alg(&l2space, num_datapoints, M, ef_construction);

  appr_alg.addPoint((void *)training_data, (size_t)0);

  for (size_t i = 1; i < num_datapoints; i++) {
    appr_alg.addPoint((void *)(training_data + vecdim * i), (size_t)i);

    if (i % 100000 == 0) {
      std::cout << i << std::endl;
    }
  }

  std::cout << "Saving index to matrix market" << std::endl;

  dump_index_to_matrix_market(appr_alg, mtx_filename.c_str(), M * 2);

  ////////////////////////////////////////////////////////////////////////
  ////////////// SEARCH //////////////////////////////////////////////////

  if (!run_search) {
    return;
  }

  size_t num_queries = 10000;
  size_t top_k = 100;
  std::vector<size_t> ef_searches{100, 200, 300, 500, 1000, 2000, 3000};

  std::vector<std::priority_queue<std::pair<float, labeltype>>> answers;
  get_gt(/* ground_truth = */ groundtruth_ids, /* queries = */ queries,
         /* training_data = */ training_data, /* qsize = */ num_queries,
         /* space = */ l2space, /* vecdim = */ vecdim,
         /* answers = */ answers, /* k = */ top_k);

  test_vs_recall(/* queries = */ queries, /* qsize = */ num_queries,
                 /* appr_alg = */ appr_alg, /* vecdim = */ vecdim,
                 /* answers = */ answers,
                 /* ef_searches = */ ef_searches, /* k = */ top_k);
}

int main(int argc, char **argv) {

  if (argc < 7) {
    std::cout << "Usage: ./dump_graph <train_data_path> <queries_path> "
                 "<groundtruth_path> <mtx_filename> <ef-construction> "
                 "<run_search>"
              << std::endl;
    std::cout << "Example: ./dump_graph "
                 "<path-to-data>/data/"
                 "sift-128-euclidean/sift-128-euclidean.train.npy "
                 "path-to-data/data/"
                 "sift-128-euclidean/sift-128-euclidean.test.npy "
                 "path-to-data/data/"
                 "sift-128-euclidean/sift-128-euclidean.gtruth.npy "
                 "sift.mtx 100 1"
              << std::endl;
    return 1;
  }

  std::string train_data_path = argv[1];
  std::string queries_path = argv[2];
  std::string groundtruth_path = argv[3];
  std::string mtx_filename = argv[4];
  int ef_construction = atoi(argv[5]);
  bool run_search = bool(atoi(argv[6]));

  run_test(/* train_data_path = */ train_data_path,
           /* queries_path = */ queries_path,
           /* groundtruth_path = */ groundtruth_path,
           /* mtx_filename = */ mtx_filename,
           /* ef_construction = */ ef_construction,
           /* run_search = */ run_search);
  return 0;
}