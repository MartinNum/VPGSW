# VPGSW
基于VP-Tree和导向小世界图（GSW）深度结合的ANNS方法
## 原理
VPGSW uses broken edge redirection algorithm instead of hierarchical structure to retain a certain percentage of long edges, thereby avoiding additional memory usage caused by hierarchical structure. Then We propose a three-stage search strategy to accelerate the search efficiency. Specifically, the first stage (searching for entry-point): We pre-build VP-Tree for the entire dataset, and then use the pre-built VP-Tree to search for the entry-point that is roughly close to the query, the second stage (fast-convergence): Using proposed VP-Tree-based guided search algorithm to efficiently converge and get the closest point to the query, the third stage (exhaustive-search): Using the range search algorithm to perform a exhaustive search to ensure the search accuracy.

## Install

#### python

在VPGSW目录下运行一下命令

```java
sudo python setup.py install
```

**python运行案例**：VPGSW/examples/python/example_angular.py

