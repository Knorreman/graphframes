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
import org.graphframes.lib.ConnectedComponents;

import java.io.Serializable;
import java.util.Objects;

/**
 * Connected Components builder. Returns a persisted and materialized dataset; callers should
 * call {@code unpersist()} when finished. Unset configuration parameters defer to Spark SQL
 * {@code spark.graphframes.*} defaults.
 */
public final class JavaConnectedComponents implements Serializable {

  private static final String ID_COL = "id";
  private static final String COMPONENT_COL = "component";
  private static final String CONNECTED_COMPONENT_COL = "connectedComponent";

  private final JavaGraphDataset<?, ?> graph;
  private String algorithm = null;
  private Integer broadcastThreshold = null;
  private Integer checkpointInterval = null;
  private Boolean useLocalCheckpoints = null;
  private StorageLevel storageLevel = null;
  private Integer maxIter = null;

  public JavaConnectedComponents(JavaGraphDataset<?, ?> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaConnectedComponents setAlgorithm(String value) {
    this.algorithm = Objects.requireNonNull(value, "algorithm must not be null");
    return this;
  }

  public JavaConnectedComponents setBroadcastThreshold(int value) {
    this.broadcastThreshold = value;
    return this;
  }

  public JavaConnectedComponents setCheckpointInterval(int value) {
    this.checkpointInterval = value;
    return this;
  }

  public JavaConnectedComponents setUseLocalCheckpoints(boolean value) {
    this.useLocalCheckpoints = value;
    return this;
  }

  public JavaConnectedComponents setIntermediateStorageLevel(StorageLevel value) {
    this.storageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  public JavaConnectedComponents maxIter(int value) {
    this.maxIter = value;
    return this;
  }

  public Dataset<ConnectedComponentResult> run() {
    ConnectedComponents cc = graph.toGraphFrame().connectedComponents();
    if (algorithm != null) {
      cc.setAlgorithm(algorithm);
    }
    if (broadcastThreshold != null) {
      cc.setBroadcastThreshold(broadcastThreshold);
    }
    if (checkpointInterval != null) {
      cc.setCheckpointInterval(checkpointInterval);
    }
    if (useLocalCheckpoints != null) {
      cc.setUseLocalCheckpoints(useLocalCheckpoints);
    }
    if (storageLevel != null) {
      cc.setIntermediateStorageLevel(storageLevel);
    }
    if (maxIter != null) {
      cc.maxIter(maxIter);
    }
    StorageLevel targetLevel = cc.getIntermediateStorageLevel();
    Dataset<Row> rawDF = cc.run();
    Dataset<ConnectedComponentResult> typed = rawDF
        .select(
            functions.col(GraphFrame.ID()).alias(ID_COL),
            functions.col(COMPONENT_COL).alias(CONNECTED_COMPONENT_COL))
        .as(ConnectedComponentResult.encoder());
    return JavaDatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, targetLevel);
  }
}
