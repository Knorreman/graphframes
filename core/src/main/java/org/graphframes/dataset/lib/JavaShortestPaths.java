/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.graphframes.dataset.lib;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.JavaGraphDataset;
import org.graphframes.lib.ShortestPaths;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Shortest Paths builder. Returns a persisted and materialized dataset; callers should call
 * {@code unpersist()} when finished.
 */
public final class JavaShortestPaths implements Serializable {

  private static final String ID_COL = "id";
  private static final String DISTANCES_COL = "distances";

  private final JavaGraphDataset<?, ?> graph;
  private List<Long> landmarks;
  private String algorithm = "graphx";
  private boolean isDirected = true;
  private boolean useLocalCheckpoints = false;
  private int checkpointInterval = 2;
  private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

  public JavaShortestPaths(JavaGraphDataset<?, ?> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaShortestPaths landmarks(List<Long> value) {
    this.landmarks = Objects.requireNonNull(value, "landmarks must not be null");
    return this;
  }

  /** Sets the algorithm ({@code "graphx"} or {@code "graphframes"}). */
  public JavaShortestPaths setAlgorithm(String value) {
    this.algorithm = Objects.requireNonNull(value, "algorithm must not be null");
    return this;
  }

  public JavaShortestPaths setIsDirected(boolean value) {
    this.isDirected = value;
    return this;
  }

  public JavaShortestPaths setUseLocalCheckpoints(boolean value) {
    this.useLocalCheckpoints = value;
    return this;
  }

  public JavaShortestPaths setCheckpointInterval(int value) {
    this.checkpointInterval = value;
    return this;
  }

  public JavaShortestPaths setIntermediateStorageLevel(StorageLevel value) {
    this.storageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  public Dataset<ShortestPathsResult> run() {
    Objects.requireNonNull(landmarks, "landmarks must be set");
    ShortestPaths sp = graph.toGraphFrame().shortestPaths();
    sp.landmarks(new ArrayList<>(landmarks));
    sp.setAlgorithm(algorithm);
    sp.setIsDirected(isDirected);
    sp.setUseLocalCheckpoints(useLocalCheckpoints);
    sp.setCheckpointInterval(checkpointInterval);
    sp.setIntermediateStorageLevel(storageLevel);
    Dataset<Row> rawDF = sp.run();
    Dataset<ShortestPathsResult> typed = rawDF
        .select(functions.col(GraphFrame.ID()).alias(ID_COL), functions.col(DISTANCES_COL))
        .as(ShortestPathsResult.encoder());
    return JavaDatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, storageLevel);
  }
}
