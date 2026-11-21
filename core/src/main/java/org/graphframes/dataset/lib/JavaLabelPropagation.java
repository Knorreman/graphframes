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
import org.graphframes.lib.LabelPropagation;

import java.io.Serializable;
import java.util.Objects;

/**
 * Label Propagation builder. Returns a persisted and materialized dataset; callers should call
 * {@code unpersist()} when finished.
 */
public final class JavaLabelPropagation implements Serializable {

  private static final String ID_COL = "id";
  private static final String LABEL_COL = "label";

  private final JavaGraphDataset<?, ?> graph;
  private Integer maxIter = null;
  private String algorithm = "graphx";
  private boolean useLocalCheckpoints = false;
  private int checkpointInterval = 2;
  private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

  public JavaLabelPropagation(JavaGraphDataset<?, ?> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaLabelPropagation maxIter(int value) {
    this.maxIter = value;
    return this;
  }

  /** Sets the algorithm ({@code "graphx"} or {@code "graphframes"}). */
  public JavaLabelPropagation setAlgorithm(String value) {
    this.algorithm = Objects.requireNonNull(value, "algorithm must not be null");
    return this;
  }

  public JavaLabelPropagation setUseLocalCheckpoints(boolean value) {
    this.useLocalCheckpoints = value;
    return this;
  }

  public JavaLabelPropagation setCheckpointInterval(int value) {
    this.checkpointInterval = value;
    return this;
  }

  public JavaLabelPropagation setIntermediateStorageLevel(StorageLevel value) {
    this.storageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  public Dataset<LabelPropagationResult> run() {
    Objects.requireNonNull(maxIter, "maxIter must be set");
    LabelPropagation lp = graph.toGraphFrame().labelPropagation();
    lp.maxIter(maxIter);
    lp.setAlgorithm(algorithm);
    lp.setUseLocalCheckpoints(useLocalCheckpoints);
    lp.setCheckpointInterval(checkpointInterval);
    lp.setIntermediateStorageLevel(storageLevel);
    Dataset<Row> rawDF = lp.run();
    // GraphFrames LPA can emit null label for vertices with zero incoming edges
    Dataset<LabelPropagationResult> typed = rawDF
        .select(
            functions.col(GraphFrame.ID()).alias(ID_COL),
            functions.coalesce(functions.col(LABEL_COL), functions.col(GraphFrame.ID()))
                .cast("long")
                .alias(LABEL_COL))
        .as(LabelPropagationResult.encoder());
    return JavaDatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, storageLevel);
  }
}
