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
import org.graphframes.lib.MaximalIndependentSet;
import scala.Tuple2;

import java.io.Serializable;
import java.util.Objects;

/**
 * Maximal Independent Set builder. Returns a persisted and materialized dataset; callers should
 * call {@code unpersist()} when finished.
 */
public final class JavaMaximalIndependentSet<VD, ED> implements Serializable {

  private static final String ID_COL = "id";
  private static final String TUPLE_1 = "_1";
  private static final String TUPLE_2 = "_2";

  private final JavaGraphDataset<VD, ED> graph;
  private boolean useLocalCheckpoints = false;
  private int checkpointInterval = 2;
  private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

  public JavaMaximalIndependentSet(JavaGraphDataset<VD, ED> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaMaximalIndependentSet<VD, ED> setUseLocalCheckpoints(boolean value) {
    this.useLocalCheckpoints = value;
    return this;
  }

  public JavaMaximalIndependentSet<VD, ED> setCheckpointInterval(int value) {
    this.checkpointInterval = value;
    return this;
  }

  public JavaMaximalIndependentSet<VD, ED> setIntermediateStorageLevel(StorageLevel value) {
    this.storageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  public Dataset<Tuple2<Long, VD>> run(long seed) {
    MaximalIndependentSet mis = graph.toGraphFrame().maximalIndependentSet();
    mis.setUseLocalCheckpoints(useLocalCheckpoints);
    mis.setCheckpointInterval(checkpointInterval);
    mis.setIntermediateStorageLevel(storageLevel);
    Dataset<Row> rawMis = mis.run(seed);
    Dataset<Row> misIds = rawMis.select(functions.col(GraphFrame.ID()).alias(ID_COL));
    Dataset<Tuple2<Long, VD>> typed = graph.vertices()
        .toDF(TUPLE_1, TUPLE_2)
        .join(misIds, functions.col(TUPLE_1).equalTo(misIds.col(ID_COL)), "left_semi")
        .as(graph.vertexEncoder());
    return JavaDatasetCacheUtils.materializeAndReleaseRaw(rawMis, typed, storageLevel);
  }
}
