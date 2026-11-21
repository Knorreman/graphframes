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
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.GraphEdge;
import org.graphframes.dataset.GraphVertex;
import org.graphframes.dataset.JavaGraphDataset;
import scala.Tuple2;
import scala.Tuple3;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Typed Java Breadth-First Search (BFS) builder for {@link JavaGraphDataset}.
 */
public final class JavaBFS<VD, ED> implements Serializable {

  private static final String ID_COL = "id";
  private static final String ATTR_COL = "attr";
  private static final String FROM_COL = "from";
  private static final String TO_COL = "to";
  private static final String PATH_COL = "path";
  private static final String LENGTH_COL = "length";
  private static final String BFS_FROM_FLAG = "_gf_jbfs_from";
  private static final String BFS_TO_FLAG = "_gf_jbfs_to";

  private final JavaGraphDataset<VD, ED> graph;
  private FilterFunction<GraphVertex<VD>> fromPred;
  private FilterFunction<GraphVertex<VD>> toPred;
  private FilterFunction<GraphEdge<ED>> edgeFilter;
  private int maxPathLength = 10;

  public JavaBFS(JavaGraphDataset<VD, ED> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaBFS<VD, ED> fromPred(FilterFunction<GraphVertex<VD>> predicate) {
    this.fromPred = Objects.requireNonNull(predicate, "fromPred must not be null");
    return this;
  }

  public JavaBFS<VD, ED> toPred(FilterFunction<GraphVertex<VD>> predicate) {
    this.toPred = Objects.requireNonNull(predicate, "toPred must not be null");
    return this;
  }

  public JavaBFS<VD, ED> edgeFilter(FilterFunction<GraphEdge<ED>> predicate) {
    this.edgeFilter = Objects.requireNonNull(predicate, "edgeFilter must not be null");
    return this;
  }

  public JavaBFS<VD, ED> maxPathLength(int value) {
    if (value < 0) {
      throw new IllegalArgumentException(
          "BFS maxPathLength must be >= 0, but was set to " + value);
    }
    this.maxPathLength = value;
    return this;
  }

  /** Executes BFS and returns shortest paths as {@code Dataset<SimplePath>}. */
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
        .toDF(GraphFrame.ID(), BFS_FROM_FLAG, BFS_TO_FLAG);

    JavaGraphDataset<VD, ED> edgeFiltered =
        edgeFilter != null ? graph.filterEdges(edgeFilter) : graph;
    Dataset<Row> eDF = edgeFiltered.edges()
        .toDF(GraphFrame.SRC(), GraphFrame.DST(), ATTR_COL)
        .select(GraphFrame.SRC(), GraphFrame.DST());

    Dataset<Row> resDF = GraphFrame.apply(vDF, eDF).bfs()
        .fromExpr(functions.col(BFS_FROM_FLAG))
        .toExpr(functions.col(BFS_TO_FLAG))
        .maxPathLength(maxPathLength)
        .run();

    List<String> cols = Arrays.asList(resDF.columns());
    if (!cols.contains(FROM_COL) || !cols.contains(TO_COL)) {
      return graph.spark().emptyDataset(SimplePath.encoder());
    }

    List<String> edgeCols = cols.stream()
        .filter(c -> c.startsWith("e") && c.length() > 1 && c.substring(1).matches("\\d+"))
        .collect(Collectors.toList());
    if (edgeCols.isEmpty()) {
      return resDF
          .select(
              functions.array(functions.col(FROM_COL + "." + ID_COL)).alias(PATH_COL),
              functions.lit(0).alias(LENGTH_COL))
          .as(SimplePath.encoder());
    }
    Column[] vIdCols = cols.stream()
        .filter(c -> c.equals(FROM_COL) || c.equals(TO_COL)
            || (c.startsWith("v") && c.length() > 1 && c.substring(1).matches("\\d+")))
        .map(c -> functions.col(c + "." + ID_COL))
        .toArray(Column[]::new);
    return resDF
        .select(
            functions.array(vIdCols).alias(PATH_COL),
            functions.lit(edgeCols.size()).alias(LENGTH_COL))
        .as(SimplePath.encoder());
  }
}
