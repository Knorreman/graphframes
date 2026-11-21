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

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.JavaGraphDataset;
import org.graphframes.lib.SVDPlusPlus;
import scala.Tuple2;
import scala.Tuple3;

import java.io.Serializable;
import java.util.Objects;

/**
 * SVD++ collaborative filtering builder. Returns a persisted and materialized dataset alongside
 * training loss; callers should call {@code unpersist()} when finished.
 */
public final class JavaSVDPlusPlus<VD, ED> implements Serializable {

  private static final String ID_COL = "id";
  private static final String ATTR_COL = "attr";
  private static final String COLUMN1_COL = "column1";
  private static final String COLUMN2_COL = "column2";
  private static final String COLUMN3_COL = "column3";
  private static final String COLUMN4_COL = "column4";

  private final JavaGraphDataset<VD, ED> graph;
  private int rank = 10;
  private Integer maxIter = null;
  private double minVal = 0.0;
  private double maxVal = 5.0;
  private double gamma1 = 0.007;
  private double gamma2 = 0.007;
  private double gamma6 = 0.005;
  private double gamma7 = 0.015;

  public JavaSVDPlusPlus(JavaGraphDataset<VD, ED> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaSVDPlusPlus<VD, ED> rank(int value) {
    this.rank = value;
    return this;
  }

  public JavaSVDPlusPlus<VD, ED> maxIter(int value) {
    this.maxIter = value;
    return this;
  }

  public JavaSVDPlusPlus<VD, ED> minValue(double value) {
    this.minVal = value;
    return this;
  }

  public JavaSVDPlusPlus<VD, ED> maxValue(double value) {
    this.maxVal = value;
    return this;
  }

  public JavaSVDPlusPlus<VD, ED> gamma1(double value) {
    this.gamma1 = value;
    return this;
  }

  public JavaSVDPlusPlus<VD, ED> gamma2(double value) {
    this.gamma2 = value;
    return this;
  }

  public JavaSVDPlusPlus<VD, ED> gamma6(double value) {
    this.gamma6 = value;
    return this;
  }

  public JavaSVDPlusPlus<VD, ED> gamma7(double value) {
    this.gamma7 = value;
    return this;
  }

  public Tuple2<Dataset<SVDPlusPlusResult>, Double> run(MapFunction<ED, Double> weightExtractor) {
    Objects.requireNonNull(weightExtractor, "weightExtractor must not be null");
    Dataset<Row> vDF = graph.vertices().toDF(GraphFrame.ID(), ATTR_COL);
    Dataset<Row> eDF = graph.edges()
        .map(
            (MapFunction<Tuple3<Long, Long, ED>, Tuple3<Long, Long, Double>>)
                e -> new Tuple3<>(e._1(), e._2(), weightExtractor.call(e._3())),
            Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.DOUBLE()))
        .toDF(GraphFrame.SRC(), GraphFrame.DST(), SVDPlusPlus.COLUMN_WEIGHT());

    SVDPlusPlus svd = GraphFrame.apply(vDF, eDF).svdPlusPlus()
        .rank(rank)
        .minValue(minVal)
        .maxValue(maxVal)
        .gamma1(gamma1)
        .gamma2(gamma2)
        .gamma6(gamma6)
        .gamma7(gamma7);
    if (maxIter != null) {
      svd.maxIter(maxIter);
    }

    Dataset<Row> rawDF = svd.run();
    Dataset<SVDPlusPlusResult> typed = rawDF
        .select(
            functions.col(GraphFrame.ID()).alias(ID_COL),
            functions.col(SVDPlusPlus.COLUMN1()).alias(COLUMN1_COL),
            functions.col(SVDPlusPlus.COLUMN2()).alias(COLUMN2_COL),
            functions.col(SVDPlusPlus.COLUMN3()).alias(COLUMN3_COL),
            functions.col(SVDPlusPlus.COLUMN4()).alias(COLUMN4_COL))
        .as(SVDPlusPlusResult.encoder());
    Dataset<SVDPlusPlusResult> persisted = JavaDatasetCacheUtils.materializeAndReleaseRaw(
        rawDF, typed, StorageLevel.MEMORY_AND_DISK());
    return new Tuple2<>(persisted, svd.loss());
  }
}
