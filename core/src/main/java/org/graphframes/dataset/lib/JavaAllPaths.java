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

import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.GraphEdge;
import org.graphframes.dataset.GraphVertex;
import org.graphframes.dataset.JavaGraphDataset;
import org.graphframes.lib.AllPaths;
import scala.Tuple2;
import scala.Tuple3;

import java.io.Serializable;
import java.util.Objects;

/**
 * Typed Java AllPaths builder for {@link JavaGraphDataset}.
 */
public final class JavaAllPaths<VD, ED> implements Serializable {

  private static final String ATTR_COL = "attr";
  private static final String PATH_COL = "path";
  private static final String LEN_COL = "len";
  private static final String LENGTH_COL = "length";
  private static final String ALL_PATHS_FROM_FLAG = "_gf_jap_from";
  private static final String ALL_PATHS_TO_FLAG = "_gf_jap_to";

  private final JavaGraphDataset<VD, ED> graph;
  private FilterFunction<GraphVertex<VD>> fromPred;
  private FilterFunction<GraphVertex<VD>> toPred;
  private FilterFunction<GraphEdge<ED>> edgeFilter;
  private int maxPathLength = 5;
  private boolean isDirected = true;
  private boolean useLocalCheckpoints = false;
  private int checkpointInterval = 2;
  private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

  public JavaAllPaths(JavaGraphDataset<VD, ED> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaAllPaths<VD, ED> fromPred(FilterFunction<GraphVertex<VD>> predicate) {
    this.fromPred = Objects.requireNonNull(predicate, "fromPred must not be null");
    return this;
  }

  public JavaAllPaths<VD, ED> toPred(FilterFunction<GraphVertex<VD>> predicate) {
    this.toPred = Objects.requireNonNull(predicate, "toPred must not be null");
    return this;
  }

  public JavaAllPaths<VD, ED> edgeFilter(FilterFunction<GraphEdge<ED>> predicate) {
    this.edgeFilter = Objects.requireNonNull(predicate, "edgeFilter must not be null");
    return this;
  }

  public JavaAllPaths<VD, ED> maxPathLength(int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(
          "AllPaths maxPathLength must be > 0, but was set to " + value);
    }
    this.maxPathLength = value;
    return this;
  }

  public JavaAllPaths<VD, ED> setIsDirected(boolean value) {
    this.isDirected = value;
    return this;
  }

  public JavaAllPaths<VD, ED> setUseLocalCheckpoints(boolean value) {
    this.useLocalCheckpoints = value;
    return this;
  }

  public JavaAllPaths<VD, ED> setCheckpointInterval(int value) {
    this.checkpointInterval = value;
    return this;
  }

  public JavaAllPaths<VD, ED> setIntermediateStorageLevel(StorageLevel value) {
    this.storageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  public Dataset<SimplePath> run() {
    Objects.requireNonNull(fromPred, "fromPred is required");
    Objects.requireNonNull(toPred, "toPred is required");
    FilterFunction<GraphVertex<VD>> fFn = fromPred;
    FilterFunction<GraphVertex<VD>> tFn = toPred;

    Dataset<Row> vDF = graph.vertices()
        .map(
            (MapFunction<Tuple2<Long, VD>, Tuple3<Long, Boolean, Boolean>>) v -> {
              GraphVertex<VD> gv = new GraphVertex<>(v._1(), v._2());
              return new Tuple3<>(v._1(), fFn.call(gv), tFn.call(gv));
            },
            Encoders.tuple(Encoders.LONG(), Encoders.BOOLEAN(), Encoders.BOOLEAN()))
        .toDF(GraphFrame.ID(), ALL_PATHS_FROM_FLAG, ALL_PATHS_TO_FLAG);

    JavaGraphDataset<VD, ED> edgeFiltered =
        edgeFilter != null ? graph.filterEdges(edgeFilter) : graph;
    Dataset<Row> eDF = edgeFiltered.edges()
        .toDF(GraphFrame.SRC(), GraphFrame.DST(), ATTR_COL)
        .select(GraphFrame.SRC(), GraphFrame.DST())
        .distinct();

    AllPaths ap = GraphFrame.apply(vDF, eDF).allPaths();
    ap.fromExpr(functions.col(ALL_PATHS_FROM_FLAG));
    ap.toExpr(functions.col(ALL_PATHS_TO_FLAG));
    ap.maxPathLength(maxPathLength);
    ap.setIsDirected(isDirected);
    ap.setUseLocalCheckpoints(useLocalCheckpoints);
    ap.setCheckpointInterval(checkpointInterval);
    ap.setIntermediateStorageLevel(storageLevel);
    return ap.run()
        .select(
            functions.col(PATH_COL),
            functions.col(LEN_COL).cast("int").alias(LENGTH_COL))
        .as(SimplePath.encoder());
  }
}
