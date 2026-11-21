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
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.functions;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.JavaGraphDataset;
import org.graphframes.lib.PageRank;
import scala.Tuple2;
import scala.Tuple3;

import java.io.Serializable;
import java.util.Objects;

/**
 * PageRank builder. Returns a persisted and materialized {@link JavaGraphDataset}; callers
 * should call {@code unpersist()} when finished.
 */
public final class JavaPageRank<VD, ED> implements Serializable {

  private static final String ATTR_COL = "attr";
  private static final String PAGERANK_COL = "pagerank";
  private static final String WEIGHT_COL = "weight";
  private static final String TUPLE_1 = "_1";
  private static final String TUPLE_2 = "_2";
  private static final String TUPLE_3 = "_3";

  private final JavaGraphDataset<VD, ED> graph;
  private double resetProb = 0.15;
  private Integer maxIter = null;
  private Double tol = null;
  private Long srcId = null;

  public JavaPageRank(JavaGraphDataset<VD, ED> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaPageRank<VD, ED> resetProbability(double value) {
    this.resetProb = value;
    return this;
  }

  public JavaPageRank<VD, ED> maxIter(int value) {
    this.maxIter = value;
    return this;
  }

  public JavaPageRank<VD, ED> tol(double value) {
    this.tol = value;
    return this;
  }

  public JavaPageRank<VD, ED> sourceId(long value) {
    this.srcId = value;
    return this;
  }

  public JavaGraphDataset<Tuple2<VD, Double>, Tuple2<ED, Double>> run() {
    PageRank pr = graph.toGraphFrame().pageRank().resetProbability(resetProb);
    if (maxIter != null) {
      pr.maxIter(maxIter);
    }
    if (tol != null) {
      pr.tol(tol);
    }
    if (srcId != null) {
      pr.sourceId(srcId);
    }
    GraphFrame resGF = pr.run();
    Encoder<Tuple2<VD, Double>> vAttrEnc =
        Encoders.tuple(graph.vertexAttrEncoder(), Encoders.DOUBLE());
    Encoder<Tuple2<ED, Double>> eAttrEnc =
        Encoders.tuple(graph.edgeAttrEncoder(), Encoders.DOUBLE());
    Dataset<Tuple2<Long, Tuple2<VD, Double>>> vv = resGF.vertices()
        .select(
            functions.col(GraphFrame.ID()).alias(TUPLE_1),
            functions.struct(
                functions.col(ATTR_COL).alias(TUPLE_1),
                functions.col(PAGERANK_COL).alias(TUPLE_2)).alias(TUPLE_2))
        .as(Encoders.tuple(Encoders.LONG(), vAttrEnc));
    Dataset<Tuple3<Long, Long, Tuple2<ED, Double>>> ee = resGF.edges()
        .select(
            functions.col(GraphFrame.SRC()).alias(TUPLE_1),
            functions.col(GraphFrame.DST()).alias(TUPLE_2),
            functions.struct(
                functions.col(ATTR_COL).alias(TUPLE_1),
                functions.col(WEIGHT_COL).alias(TUPLE_2)).alias(TUPLE_3))
        .as(Encoders.tuple(Encoders.LONG(), Encoders.LONG(), eAttrEnc));
    return JavaDatasetCacheUtils.materializeGraphAndReleaseRaw(
        resGF, vv, ee, vAttrEnc, eAttrEnc, StorageLevel.MEMORY_AND_DISK());
  }
}
