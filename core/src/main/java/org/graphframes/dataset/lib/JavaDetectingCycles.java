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
import org.graphframes.dataset.JavaGraphDataset;
import org.graphframes.lib.DetectingCycles;

import java.io.Serializable;
import java.util.Objects;

/**
 * Cycle Detection builder. Returns a persisted and materialized dataset; callers should call
 * {@code unpersist()} when finished.
 */
public final class JavaDetectingCycles implements Serializable {

  private static final String PATH_COL = "path";
  private static final String LENGTH_COL = "length";

  private final JavaGraphDataset<?, ?> graph;
  private boolean useLocalCheckpoints = false;
  private int checkpointInterval = 2;
  private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

  public JavaDetectingCycles(JavaGraphDataset<?, ?> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaDetectingCycles setUseLocalCheckpoints(boolean value) {
    this.useLocalCheckpoints = value;
    return this;
  }

  public JavaDetectingCycles setCheckpointInterval(int value) {
    this.checkpointInterval = value;
    return this;
  }

  public JavaDetectingCycles setIntermediateStorageLevel(StorageLevel value) {
    this.storageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  public Dataset<SimplePath> run() {
    DetectingCycles dc = graph.toGraphFrame().detectingCycles();
    dc.setUseLocalCheckpoints(useLocalCheckpoints);
    dc.setCheckpointInterval(checkpointInterval);
    dc.setIntermediateStorageLevel(storageLevel);
    Dataset<Row> rawDF = dc.run();
    Dataset<SimplePath> typed = rawDF
        .select(
            functions.col(DetectingCycles.foundSeqCol()).alias(PATH_COL),
            functions.size(functions.col(DetectingCycles.foundSeqCol())).minus(1)
                .alias(LENGTH_COL))
        .as(SimplePath.encoder());
    return JavaDatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, storageLevel);
  }
}
