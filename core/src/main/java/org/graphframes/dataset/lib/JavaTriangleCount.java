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
import org.graphframes.lib.TriangleCount;

import java.io.Serializable;
import java.util.Objects;

/**
 * Triangle Count builder. Returns a persisted and materialized dataset; callers should call
 * {@code unpersist()} when finished.
 */
public final class JavaTriangleCount implements Serializable {

  private static final String ID_COL = "id";
  private static final String COUNT_COL = "count";

  private final JavaGraphDataset<?, ?> graph;
  private String algorithm = "exact";
  private int lgNomEntries = 12;
  private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

  public JavaTriangleCount(JavaGraphDataset<?, ?> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaTriangleCount setAlgorithm(String value) {
    Objects.requireNonNull(value, "algorithm must not be null");
    if (!"exact".equals(value) && !"approx".equals(value)) {
      throw new IllegalArgumentException(
          "supported algorithms: exact, approx, but got: " + value);
    }
    this.algorithm = value;
    return this;
  }

  /** Sets log2 of nominal entries used by approximate Theta sketches (4 to 26). */
  public JavaTriangleCount setLgNomEntries(int value) {
    if (value < 4 || value > 26) {
      throw new IllegalArgumentException(
          "lgNomEntries must be between 4 and 26, but got " + value);
    }
    this.lgNomEntries = value;
    return this;
  }

  public JavaTriangleCount setIntermediateStorageLevel(StorageLevel value) {
    this.storageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  public Dataset<TriangleCountResult> run() {
    TriangleCount tc = graph.toGraphFrame().triangleCount();
    tc.setAlgorithm(algorithm);
    tc.setLgNomEntries(lgNomEntries);
    tc.setIntermediateStorageLevel(storageLevel);
    Dataset<Row> rawDF = tc.run();
    Dataset<TriangleCountResult> typed = rawDF
        .select(
            functions.col(GraphFrame.ID()).alias(ID_COL),
            functions.col(COUNT_COL).cast("long").alias(COUNT_COL))
        .as(TriangleCountResult.encoder());
    return JavaDatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, storageLevel);
  }
}
