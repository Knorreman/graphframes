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

package org.graphframes;

import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.lib.ClusterAssignment;
import org.graphframes.lib.ConnectedComponentResult;
import org.graphframes.lib.ConnectedComponents;
import org.graphframes.lib.DetectingCycles;
import org.graphframes.lib.GraphEdge;
import org.graphframes.lib.GraphTriplet;
import org.graphframes.lib.GraphVertex;
import org.graphframes.lib.JavaAggregateMessages;
import org.graphframes.lib.JavaPregel;
import org.graphframes.lib.KCore;
import org.graphframes.lib.KCoreResult;
import org.graphframes.lib.LabelPropagationResult;
import org.graphframes.lib.SVDPlusPlus;
import org.graphframes.lib.SVDPlusPlusResult;
import org.graphframes.lib.ShortestPathsResult;
import org.graphframes.lib.SimplePath;
import org.graphframes.lib.TriangleCountResult;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;
import scala.Tuple5;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Java-friendly typed graph API backed by Spark {@code Dataset}s.
 *
 * <p>Vertices are represented as {@code Dataset<Tuple2<Long, VD>>} and edges as
 * {@code Dataset<Tuple3<Long, Long, ED>>}, using explicit Spark {@code Encoder}s (such as
 * {@code Encoders.STRING()}, {@code Encoders.INT()}, or {@code Encoders.bean(Class)}) so that
 * Java callers do not need Scala {@code TypeTag}s or implicit encoders.
 *
 * @param <VD> the vertex attribute type
 * @param <ED> the edge attribute type
 */
public class JavaGraphDataset<VD, ED> implements Serializable {

  private static final String ATTR_COL = "attr";

  private final transient Dataset<Tuple2<Long, VD>> vertices;
  private final transient Dataset<Tuple3<Long, Long, ED>> edges;
  private final Encoder<VD> vdEncoder;
  private final Encoder<ED> edEncoder;

  private JavaGraphDataset(
      Dataset<Tuple2<Long, VD>> vertices,
      Dataset<Tuple3<Long, Long, ED>> edges,
      Encoder<VD> vdEncoder,
      Encoder<ED> edEncoder) {
    this.vertices = Objects.requireNonNull(vertices, "vertices must not be null");
    this.edges = Objects.requireNonNull(edges, "edges must not be null");
    this.vdEncoder = Objects.requireNonNull(vdEncoder, "vdEncoder must not be null");
    this.edEncoder = Objects.requireNonNull(edEncoder, "edEncoder must not be null");
  }

  /**
   * Constructs a {@link JavaGraphDataset} from typed vertex and edge datasets.
   */
  public static <VD, ED> JavaGraphDataset<VD, ED> of(
      Dataset<Tuple2<Long, VD>> vertices,
      Dataset<Tuple3<Long, Long, ED>> edges,
      Encoder<VD> vdEncoder,
      Encoder<ED> edEncoder) {
    return new JavaGraphDataset<>(vertices, edges, vdEncoder, edEncoder);
  }

  /**
   * Constructs a {@link JavaGraphDataset} from an edge dataset, inferring distinct vertex IDs
   * from edge endpoints and assigning {@code defaultVertexAttr} to each vertex.
   */
  public static <VD, ED> JavaGraphDataset<VD, ED> fromEdges(
      Dataset<Tuple3<Long, Long, ED>> edges,
      VD defaultVertexAttr,
      Encoder<VD> vdEncoder,
      Encoder<ED> edEncoder) {
    Dataset<Long> ids = edges
        .flatMap(
            (FlatMapFunction<Tuple3<Long, Long, ED>, Long>)
                e -> Arrays.asList(e._1(), e._2()).iterator(),
            Encoders.LONG())
        .distinct();
    Dataset<Tuple2<Long, VD>> vertices = ids.map(
        (MapFunction<Long, Tuple2<Long, VD>>) id -> new Tuple2<>(id, defaultVertexAttr),
        Encoders.tuple(Encoders.LONG(), vdEncoder));
    return new JavaGraphDataset<>(vertices, edges, vdEncoder, edEncoder);
  }

  /**
   * Constructs a {@link JavaGraphDataset} from an existing {@link GraphFrame} whose vertices have
   * {@code id} and {@code attr} columns and whose edges have {@code src}, {@code dst}, and
   * {@code attr} columns.
   */
  public static <VD, ED> JavaGraphDataset<VD, ED> fromGraphFrame(
      GraphFrame graph,
      Encoder<VD> vdEncoder,
      Encoder<ED> edEncoder) {
    Encoder<Tuple2<Long, VD>> vEnc = Encoders.tuple(Encoders.LONG(), vdEncoder);
    Encoder<Tuple3<Long, Long, ED>> eEnc =
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), edEncoder);
    Dataset<Tuple2<Long, VD>> vv = graph.vertices()
        .select(
            functions.col(GraphFrame.ID()).cast("long").alias("_1"),
            functions.col(ATTR_COL).alias("_2"))
        .as(vEnc);
    Dataset<Tuple3<Long, Long, ED>> ee = graph.edges()
        .select(
            functions.col(GraphFrame.SRC()).cast("long").alias("_1"),
            functions.col(GraphFrame.DST()).cast("long").alias("_2"),
            functions.col(ATTR_COL).alias("_3"))
        .as(eEnc);
    return new JavaGraphDataset<>(vv, ee, vdEncoder, edEncoder);
  }

  public SparkSession spark() {
    return vertices().sparkSession();
  }

  public Encoder<VD> vertexAttrEncoder() {
    return vdEncoder;
  }

  public Encoder<ED> edgeAttrEncoder() {
    return edEncoder;
  }

  public Encoder<Tuple2<Long, VD>> vertexEncoder() {
    return Encoders.tuple(Encoders.LONG(), vdEncoder);
  }

  public Encoder<Tuple3<Long, Long, ED>> edgeEncoder() {
    return Encoders.tuple(Encoders.LONG(), Encoders.LONG(), edEncoder);
  }

  public Encoder<Tuple5<Long, Long, VD, VD, ED>> tripletEncoder() {
    return Encoders.tuple(Encoders.LONG(), Encoders.LONG(), vdEncoder, vdEncoder, edEncoder);
  }

  public Dataset<Tuple2<Long, VD>> vertices() {
    if (vertices == null) {
      throw new IllegalStateException(
          "You cannot use JavaGraphDataset objects within a Spark closure");
    }
    return vertices;
  }

  public Dataset<Tuple3<Long, Long, ED>> edges() {
    if (edges == null) {
      throw new IllegalStateException(
          "You cannot use JavaGraphDataset objects within a Spark closure");
    }
    return edges;
  }

  /**
   * Returns typed triplets {@code (srcId, dstId, srcAttr, dstAttr, edgeAttr)} for all edges
   * whose endpoints exist in {@code vertices()}.
   */
  public Dataset<Tuple5<Long, Long, VD, VD, ED>> triplets() {
    Dataset<Row> srcV = vertices().toDF("srcId", "srcAttr");
    Dataset<Row> dstV = vertices().toDF("dstId", "dstAttr");
    return edges()
        .toDF("srcId", "dstId", "attr")
        .join(srcV, "srcId")
        .join(dstV, "dstId")
        .select(
            functions.col("srcId").alias("_1"),
            functions.col("dstId").alias("_2"),
            functions.col("srcAttr").alias("_3"),
            functions.col("dstAttr").alias("_4"),
            functions.col("attr").alias("_5"))
        .as(tripletEncoder());
  }

  @Override
  public String toString() {
    return "JavaGraphDataset(v:" + vertices().toString() + ", e:" + edges().toString() + ")";
  }

  public JavaGraphDataset<VD, ED> cache() {
    return persist();
  }

  public JavaGraphDataset<VD, ED> persist() {
    vertices().persist();
    edges().persist();
    return this;
  }

  public JavaGraphDataset<VD, ED> persist(StorageLevel newLevel) {
    vertices().persist(newLevel);
    edges().persist(newLevel);
    return this;
  }

  public JavaGraphDataset<VD, ED> unpersist() {
    vertices().unpersist();
    edges().unpersist();
    return this;
  }

  public JavaGraphDataset<VD, ED> unpersist(boolean blocking) {
    vertices().unpersist(blocking);
    edges().unpersist(blocking);
    return this;
  }

  /** Converts this {@link JavaGraphDataset} into a {@link GraphFrame}. */
  public GraphFrame toGraphFrame() {
    Dataset<Row> vDF = vertices().toDF(GraphFrame.ID(), ATTR_COL);
    Dataset<Row> eDF = edges().toDF(GraphFrame.SRC(), GraphFrame.DST(), ATTR_COL);
    return GraphFrame.apply(vDF, eDF);
  }

  /** Validates graph consistency (duplicate vertices and dangling edge endpoints). */
  public void validate() {
    toGraphFrame().validate();
  }

  /** Validates graph consistency with custom options. */
  public void validate(boolean checkVertices, StorageLevel intermediateStorageLevel) {
    toGraphFrame().validate(checkVertices, intermediateStorageLevel);
  }

  /** Returns a new graph with all edge directions reversed. */
  public JavaGraphDataset<VD, ED> asReversed() {
    Dataset<Tuple3<Long, Long, ED>> rev = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple3<Long, Long, ED>>)
            e -> new Tuple3<>(e._2(), e._1(), e._3()),
        edgeEncoder());
    return new JavaGraphDataset<>(vertices(), rev, vdEncoder, edEncoder);
  }

  /** Returns a new undirected graph adding reverse edges for every directed edge. */
  public JavaGraphDataset<VD, ED> asUndirected() {
    Dataset<Tuple3<Long, Long, ED>> rev = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple3<Long, Long, ED>>)
            e -> new Tuple3<>(e._2(), e._1(), e._3()),
        edgeEncoder());
    return new JavaGraphDataset<>(vertices(), edges().union(rev), vdEncoder, edEncoder);
  }

  // ============== Degree metrics ==============

  public Dataset<Tuple2<Long, Integer>> outDegrees() {
    return edges()
        .toDF("srcId", "dstId", "attr")
        .groupBy(functions.col("srcId"))
        .agg(functions.count("*").cast("int"))
        .toDF("_1", "_2")
        .as(Encoders.tuple(Encoders.LONG(), Encoders.INT()));
  }

  public Dataset<Tuple2<Long, Integer>> inDegrees() {
    return edges()
        .toDF("srcId", "dstId", "attr")
        .groupBy(functions.col("dstId"))
        .agg(functions.count("*").cast("int"))
        .toDF("_1", "_2")
        .as(Encoders.tuple(Encoders.LONG(), Encoders.INT()));
  }

  public Dataset<Tuple2<Long, Integer>> degrees() {
    return edges()
        .toDF("srcId", "dstId", "attr")
        .select(functions.explode(functions.array(functions.col("srcId"), functions.col("dstId")))
            .alias("id"))
        .groupBy(functions.col("id"))
        .agg(functions.count("*").cast("int"))
        .toDF("_1", "_2")
        .as(Encoders.tuple(Encoders.LONG(), Encoders.INT()));
  }

  /**
   * Computes per-edge-type out-degrees as {@code Dataset<Tuple3<Long, K, Integer>>}
   * {@code (vertexId, edgeType, count)}.
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeOutDegree(
      MapFunction<ED, K> edgeTypeExtractor, Encoder<K> keyEncoder) {
    Dataset<Tuple2<Long, K>> pairs = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>)
            e -> new Tuple2<>(e._1(), edgeTypeExtractor.call(e._3())),
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, keyEncoder);
  }

  /**
   * Computes per-edge-type in-degrees as {@code Dataset<Tuple3<Long, K, Integer>>}
   * {@code (vertexId, edgeType, count)}.
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeInDegree(
      MapFunction<ED, K> edgeTypeExtractor, Encoder<K> keyEncoder) {
    Dataset<Tuple2<Long, K>> pairs = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>)
            e -> new Tuple2<>(e._2(), edgeTypeExtractor.call(e._3())),
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, keyEncoder);
  }

  /**
   * Computes per-edge-type total degrees as {@code Dataset<Tuple3<Long, K, Integer>>}
   * {@code (vertexId, edgeType, count)}.
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeDegree(
      MapFunction<ED, K> edgeTypeExtractor, Encoder<K> keyEncoder) {
    Dataset<Tuple2<Long, K>> pairs = edges().flatMap(
        (FlatMapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>) e -> {
          K k = edgeTypeExtractor.call(e._3());
          return Arrays.asList(new Tuple2<>(e._1(), k), new Tuple2<>(e._2(), k)).iterator();
        },
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, keyEncoder);
  }

  private <K> Dataset<Tuple3<Long, K, Integer>> countByKey(
      Dataset<Tuple2<Long, K>> pairs, Encoder<K> keyEncoder) {
    return pairs
        .toDF("id", "k")
        .groupBy(functions.col("id"), functions.col("k"))
        .agg(functions.count("*").cast("int").alias("cnt"))
        .select(
            functions.col("id").alias("_1"),
            functions.col("k").alias("_2"),
            functions.col("cnt").alias("_3"))
        .as(Encoders.tuple(Encoders.LONG(), keyEncoder, Encoders.INT()));
  }

  // ============== Subgraphs & transformations ==============

  /** Filters vertices by predicate and removes edges referencing dropped vertices. */
  public JavaGraphDataset<VD, ED> filterVertices(FilterFunction<GraphVertex<VD>> condition) {
    Dataset<Tuple2<Long, VD>> vv = vertices().filter(
        (FilterFunction<Tuple2<Long, VD>>) v -> condition.call(new GraphVertex<>(v._1(), v._2())));
    Dataset<Row> vvIds = vv.toDF("id", "attr").select(functions.col("id"));
    Dataset<Tuple3<Long, Long, ED>> ee = edges()
        .toDF("_1", "_2", "_3")
        .join(vvIds, functions.col("_1").equalTo(vvIds.col("id")), "left_semi")
        .join(vvIds, functions.col("_2").equalTo(vvIds.col("id")), "left_semi")
        .as(edgeEncoder());
    return new JavaGraphDataset<>(vv, ee, vdEncoder, edEncoder);
  }

  /** Filters edges by predicate while keeping all vertices. */
  public JavaGraphDataset<VD, ED> filterEdges(FilterFunction<GraphEdge<ED>> condition) {
    Dataset<Tuple3<Long, Long, ED>> ee = edges().filter(
        (FilterFunction<Tuple3<Long, Long, ED>>)
            e -> condition.call(new GraphEdge<>(e._1(), e._2(), e._3())));
    return new JavaGraphDataset<>(vertices(), ee, vdEncoder, edEncoder);
  }

  /** Restricts the graph to vertices matching {@code vpred} and edges matching {@code epred}. */
  public JavaGraphDataset<VD, ED> subgraph(
      FilterFunction<GraphEdge<ED>> epred, FilterFunction<GraphVertex<VD>> vpred) {
    return filterVertices(vpred).filterEdges(epred);
  }

  /** Drops vertices that are not incident to any edge. */
  public JavaGraphDataset<VD, ED> dropIsolatedVertices() {
    Dataset<Row> activeIds = edges()
        .toDF("srcId", "dstId", "attr")
        .select(functions.explode(functions.array(functions.col("srcId"), functions.col("dstId")))
            .alias("id"));
    Dataset<Tuple2<Long, VD>> vv = vertices()
        .toDF("_1", "_2")
        .join(activeIds, functions.col("_1").equalTo(activeIds.col("id")), "left_semi")
        .as(vertexEncoder());
    return new JavaGraphDataset<>(vv, edges(), vdEncoder, edEncoder);
  }

  /** Transforms each vertex attribute. */
  public <VD2> JavaGraphDataset<VD2, ED> mapVertices(
      Function2<Long, VD, VD2> fn, Encoder<VD2> vd2Encoder) {
    Dataset<Tuple2<Long, VD2>> vv = vertices().map(
        (MapFunction<Tuple2<Long, VD>, Tuple2<Long, VD2>>)
            v -> new Tuple2<>(v._1(), fn.call(v._1(), v._2())),
        Encoders.tuple(Encoders.LONG(), vd2Encoder));
    return new JavaGraphDataset<>(vv, edges(), vd2Encoder, edEncoder);
  }

  /** Transforms each edge attribute. */
  public <ED2> JavaGraphDataset<VD, ED2> mapEdges(
      MapFunction<GraphEdge<ED>, ED2> fn, Encoder<ED2> ed2Encoder) {
    Dataset<Tuple3<Long, Long, ED2>> ee = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple3<Long, Long, ED2>>)
            e -> new Tuple3<>(e._1(), e._2(), fn.call(new GraphEdge<>(e._1(), e._2(), e._3()))),
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), ed2Encoder));
    return new JavaGraphDataset<>(vertices(), ee, vdEncoder, ed2Encoder);
  }

  /** Transforms each edge attribute using the full triplet. */
  public <ED2> JavaGraphDataset<VD, ED2> mapTriplets(
      MapFunction<GraphTriplet<VD, ED>, ED2> fn, Encoder<ED2> ed2Encoder) {
    Dataset<Tuple3<Long, Long, ED2>> ee = triplets().map(
        (MapFunction<Tuple5<Long, Long, VD, VD, ED>, Tuple3<Long, Long, ED2>>)
            t -> new Tuple3<>(
                t._1(),
                t._2(),
                fn.call(new GraphTriplet<>(t._1(), t._2(), t._3(), t._4(), t._5()))),
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), ed2Encoder));
    return new JavaGraphDataset<>(vertices(), ee, vdEncoder, ed2Encoder);
  }

  // ============== Message passing & Pregel ==============

  /** Returns a typed Java message-aggregation builder for message type {@code M}. */
  public <M> JavaAggregateMessages<VD, ED, M> aggregateMessages(Encoder<M> msgEncoder) {
    return new JavaAggregateMessages<>(this, msgEncoder);
  }

  /** Returns a typed Java Pregel builder. */
  public JavaPregel<VD, ED> pregel() {
    return new JavaPregel<>(this);
  }

  // ============== Standard graph algorithms ==============

  /** Connected components algorithm returning {@code Dataset<ConnectedComponentResult>}. */
  public JavaConnectedComponents connectedComponents() {
    return new JavaConnectedComponents(this);
  }

  /** PageRank algorithm returning a {@link JavaGraphDataset} with PageRank & edge weights. */
  public JavaPageRank<VD, ED> pageRank() {
    return new JavaPageRank<>(this);
  }

  /** Shortest paths algorithm returning {@code Dataset<ShortestPathsResult>}. */
  public JavaShortestPaths shortestPaths() {
    return new JavaShortestPaths(this);
  }

  /** Triangle count algorithm returning {@code Dataset<TriangleCountResult>}. */
  public JavaTriangleCount triangleCount() {
    return new JavaTriangleCount(this);
  }

  /** Label propagation algorithm returning {@code Dataset<LabelPropagationResult>}. */
  public JavaLabelPropagation labelPropagation() {
    return new JavaLabelPropagation(this);
  }

  /** Strongly connected components returning {@code Dataset<ConnectedComponentResult>}. */
  public JavaStronglyConnectedComponents stronglyConnectedComponents() {
    return new JavaStronglyConnectedComponents(this);
  }

  /** K-Core decomposition returning {@code Dataset<KCoreResult>}. */
  public JavaKCore kCore() {
    return new JavaKCore(this);
  }

  /** Maximal Independent Set returning {@code Dataset<Tuple2<Long, VD>>}. */
  public JavaMaximalIndependentSet<VD, ED> maximalIndependentSet() {
    return new JavaMaximalIndependentSet<>(this);
  }

  /** Cycle detection returning {@code Dataset<SimplePath>}. */
  public JavaDetectingCycles detectingCycles() {
    return new JavaDetectingCycles(this);
  }

  /** Typed Breadth-First Search returning {@code Dataset<SimplePath>}. */
  public JavaBFS<VD, ED> bfs() {
    return new JavaBFS<>(this);
  }

  /** Typed AllPaths enumerating all simple paths returning {@code Dataset<SimplePath>}. */
  public JavaAllPaths<VD, ED> allPaths() {
    return new JavaAllPaths<>(this);
  }

  /** SVD++ collaborative filtering with a typed edge rating extractor. */
  public JavaSVDPlusPlus<VD, ED> svdPlusPlus() {
    return new JavaSVDPlusPlus<>(this);
  }

  /** Power Iteration Clustering without edge weights. */
  public Dataset<ClusterAssignment> powerIterationClustering(int k, int maxIter) {
    return toGraphFrame()
        .powerIterationClustering(k, maxIter, scala.Option.empty())
        .select(
            functions.col(GraphFrame.ID()).alias("id"),
            functions.col("cluster").cast("int").alias("cluster"))
        .as(ClusterAssignment.encoder());
  }

  /** Power Iteration Clustering with a typed edge weight extractor. */
  public Dataset<ClusterAssignment> powerIterationClustering(
      int k, int maxIter, MapFunction<ED, Double> weightExtractor) {
    String weightCol = "_gf_pic_weight";
    Dataset<Row> vDF = vertices().toDF(GraphFrame.ID(), ATTR_COL);
    Dataset<Row> eDF = edges()
        .map(
            (MapFunction<Tuple3<Long, Long, ED>, Tuple3<Long, Long, Double>>)
                e -> new Tuple3<>(e._1(), e._2(), weightExtractor.call(e._3())),
            Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.DOUBLE()))
        .toDF(GraphFrame.SRC(), GraphFrame.DST(), weightCol);
    return GraphFrame.apply(vDF, eDF)
        .powerIterationClustering(k, maxIter, scala.Option.apply(weightCol))
        .select(
            functions.col(GraphFrame.ID()).alias("id"),
            functions.col("cluster").cast("int").alias("cluster"))
        .as(ClusterAssignment.encoder());
  }

  // ============== Nested Algorithm Builders ==============

  public static final class JavaConnectedComponents implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private String algorithm = ConnectedComponents.ALGO_TWO_PHASE();
    private int broadcastThreshold = 1000000;
    private int checkpointInterval = 2;
    private boolean useLocalCheckpoints = false;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();
    private Integer maxIter = null;

    JavaConnectedComponents(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaConnectedComponents setAlgorithm(String value) {
      this.algorithm = value;
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
      this.storageLevel = value;
      return this;
    }

    public JavaConnectedComponents maxIter(int value) {
      this.maxIter = value;
      return this;
    }

    public Dataset<ConnectedComponentResult> run() {
      ConnectedComponents cc = graph.toGraphFrame().connectedComponents();
      cc.setAlgorithm(algorithm);
      cc.setBroadcastThreshold(broadcastThreshold);
      cc.setCheckpointInterval(checkpointInterval);
      cc.setUseLocalCheckpoints(useLocalCheckpoints);
      cc.setIntermediateStorageLevel(storageLevel);
      if (maxIter != null) {
        cc.maxIter(maxIter);
      }
      return cc.run()
          .select(
              functions.col(GraphFrame.ID()).alias("id"),
              functions.col("component").alias("connectedComponent"))
          .as(ConnectedComponentResult.encoder());
    }
  }

  public static final class JavaPageRank<VD, ED> implements Serializable {
    private final JavaGraphDataset<VD, ED> graph;
    private double resetProb = 0.15;
    private Integer maxIter = null;
    private Double tol = null;
    private Long srcId = null;

    JavaPageRank(JavaGraphDataset<VD, ED> graph) {
      this.graph = graph;
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
      org.graphframes.lib.PageRank pr =
          graph.toGraphFrame().pageRank().resetProbability(resetProb);
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
              functions.col(GraphFrame.ID()).alias("_1"),
              functions.struct(
                  functions.col(ATTR_COL).alias("_1"),
                  functions.col("pagerank").alias("_2")).alias("_2"))
          .as(Encoders.tuple(Encoders.LONG(), vAttrEnc));
      Dataset<Tuple3<Long, Long, Tuple2<ED, Double>>> ee = resGF.edges()
          .select(
              functions.col(GraphFrame.SRC()).alias("_1"),
              functions.col(GraphFrame.DST()).alias("_2"),
              functions.struct(
                  functions.col(ATTR_COL).alias("_1"),
                  functions.col("weight").alias("_2")).alias("_3"))
          .as(Encoders.tuple(Encoders.LONG(), Encoders.LONG(), eAttrEnc));
      return JavaGraphDataset.of(vv, ee, vAttrEnc, eAttrEnc);
    }
  }

  public static final class JavaShortestPaths implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private List<Long> landmarks;
    private boolean isDirected = true;
    private boolean useLocalCheckpoints = false;
    private int checkpointInterval = 2;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaShortestPaths(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaShortestPaths landmarks(List<Long> value) {
      this.landmarks = value;
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
      this.storageLevel = value;
      return this;
    }

    public Dataset<ShortestPathsResult> run() {
      Objects.requireNonNull(landmarks, "landmarks must be set");
      org.graphframes.lib.ShortestPaths sp = graph.toGraphFrame().shortestPaths();
      sp.landmarks(new ArrayList<>(landmarks));
      sp.setIsDirected(isDirected);
      sp.setUseLocalCheckpoints(useLocalCheckpoints);
      sp.setCheckpointInterval(checkpointInterval);
      sp.setIntermediateStorageLevel(storageLevel);
      return sp.run()
          .select(functions.col(GraphFrame.ID()).alias("id"), functions.col("distances"))
          .as(ShortestPathsResult.encoder());
    }
  }

  public static final class JavaTriangleCount implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private String algorithm = "exact";
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaTriangleCount(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaTriangleCount setAlgorithm(String value) {
      this.algorithm = value;
      return this;
    }

    public JavaTriangleCount setIntermediateStorageLevel(StorageLevel value) {
      this.storageLevel = value;
      return this;
    }

    public Dataset<TriangleCountResult> run() {
      org.graphframes.lib.TriangleCount tc = graph.toGraphFrame().triangleCount();
      tc.setAlgorithm(algorithm);
      tc.setIntermediateStorageLevel(storageLevel);
      return tc.run()
          .select(
              functions.col(GraphFrame.ID()).alias("id"),
              functions.col("count").cast("long").alias("count"))
          .as(TriangleCountResult.encoder());
    }
  }

  public static final class JavaLabelPropagation implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private Integer maxIter = null;
    private boolean useLocalCheckpoints = false;
    private int checkpointInterval = 2;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaLabelPropagation(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaLabelPropagation maxIter(int value) {
      this.maxIter = value;
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
      this.storageLevel = value;
      return this;
    }

    public Dataset<LabelPropagationResult> run() {
      Objects.requireNonNull(maxIter, "maxIter must be set");
      org.graphframes.lib.LabelPropagation lp = graph.toGraphFrame().labelPropagation();
      lp.maxIter(maxIter);
      lp.setUseLocalCheckpoints(useLocalCheckpoints);
      lp.setCheckpointInterval(checkpointInterval);
      lp.setIntermediateStorageLevel(storageLevel);
      return lp.run()
          .select(
              functions.col(GraphFrame.ID()).alias("id"),
              functions.col("label").cast("long").alias("label"))
          .as(LabelPropagationResult.encoder());
    }
  }

  public static final class JavaStronglyConnectedComponents implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private Integer maxIter = null;

    JavaStronglyConnectedComponents(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaStronglyConnectedComponents maxIter(int value) {
      this.maxIter = value;
      return this;
    }

    public Dataset<ConnectedComponentResult> run() {
      Objects.requireNonNull(maxIter, "maxIter must be set");
      org.graphframes.lib.StronglyConnectedComponents scc =
          graph.toGraphFrame().stronglyConnectedComponents();
      scc.maxIter(maxIter);
      return scc.run()
          .select(
              functions.col(GraphFrame.ID()).alias("id"),
              functions.col("component").alias("connectedComponent"))
          .as(ConnectedComponentResult.encoder());
    }
  }

  public static final class JavaKCore implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private boolean useLocalCheckpoints = false;
    private int checkpointInterval = 2;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaKCore(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaKCore setUseLocalCheckpoints(boolean value) {
      this.useLocalCheckpoints = value;
      return this;
    }

    public JavaKCore setCheckpointInterval(int value) {
      this.checkpointInterval = value;
      return this;
    }

    public JavaKCore setIntermediateStorageLevel(StorageLevel value) {
      this.storageLevel = value;
      return this;
    }

    public Dataset<KCoreResult> run() {
      KCore kc = graph.toGraphFrame().kCore();
      kc.setUseLocalCheckpoints(useLocalCheckpoints);
      kc.setCheckpointInterval(checkpointInterval);
      kc.setIntermediateStorageLevel(storageLevel);
      return kc.run()
          .select(
              functions.col(GraphFrame.ID()).alias("id"),
              functions.col(KCore.kCoreColumnName()).alias("kcore"))
          .as(KCoreResult.encoder());
    }
  }

  public static final class JavaMaximalIndependentSet<VD, ED> implements Serializable {
    private final JavaGraphDataset<VD, ED> graph;
    private boolean useLocalCheckpoints = false;
    private int checkpointInterval = 2;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaMaximalIndependentSet(JavaGraphDataset<VD, ED> graph) {
      this.graph = graph;
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
      this.storageLevel = value;
      return this;
    }

    public Dataset<Tuple2<Long, VD>> run(long seed) {
      org.graphframes.lib.MaximalIndependentSet mis =
          graph.toGraphFrame().maximalIndependentSet();
      mis.setUseLocalCheckpoints(useLocalCheckpoints);
      mis.setCheckpointInterval(checkpointInterval);
      mis.setIntermediateStorageLevel(storageLevel);
      Dataset<Row> misIds = mis.run(seed)
          .select(functions.col(GraphFrame.ID()).alias("id"));
      return graph.vertices()
          .toDF("_1", "_2")
          .join(misIds, functions.col("_1").equalTo(misIds.col("id")), "left_semi")
          .as(graph.vertexEncoder());
    }
  }

  public static final class JavaDetectingCycles implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private boolean useLocalCheckpoints = false;
    private int checkpointInterval = 2;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaDetectingCycles(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
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
      this.storageLevel = value;
      return this;
    }

    public Dataset<SimplePath> run() {
      DetectingCycles dc = graph.toGraphFrame().detectingCycles();
      dc.setUseLocalCheckpoints(useLocalCheckpoints);
      dc.setCheckpointInterval(checkpointInterval);
      dc.setIntermediateStorageLevel(storageLevel);
      return dc.run()
          .select(
              functions.col(DetectingCycles.foundSeqCol()).alias("path"),
              functions.size(functions.col(DetectingCycles.foundSeqCol())).minus(1).alias("length"))
          .as(SimplePath.encoder());
    }
  }

  public static final class JavaBFS<VD, ED> implements Serializable {
    private final JavaGraphDataset<VD, ED> graph;
    private FilterFunction<GraphVertex<VD>> fromPred;
    private FilterFunction<GraphVertex<VD>> toPred;
    private FilterFunction<GraphEdge<ED>> edgeFilter;
    private int maxPathLength = 10;

    JavaBFS(JavaGraphDataset<VD, ED> graph) {
      this.graph = graph;
    }

    public JavaBFS<VD, ED> fromPred(FilterFunction<GraphVertex<VD>> predicate) {
      this.fromPred = predicate;
      return this;
    }

    public JavaBFS<VD, ED> toPred(FilterFunction<GraphVertex<VD>> predicate) {
      this.toPred = predicate;
      return this;
    }

    public JavaBFS<VD, ED> edgeFilter(FilterFunction<GraphEdge<ED>> predicate) {
      this.edgeFilter = predicate;
      return this;
    }

    public JavaBFS<VD, ED> maxPathLength(int value) {
      this.maxPathLength = value;
      return this;
    }

    /** Executes BFS and returns shortest paths as {@code Dataset<SimplePath>}. */
    public Dataset<SimplePath> run() {
      Objects.requireNonNull(fromPred, "fromPred is required");
      Objects.requireNonNull(toPred, "toPred is required");
      FilterFunction<GraphVertex<VD>> fFn = fromPred;
      FilterFunction<GraphVertex<VD>> tFn = toPred;

      String fromCol = "_gf_jbfs_from";
      String toCol = "_gf_jbfs_to";
      Dataset<Row> vDF = graph.vertices()
          .map(
              (MapFunction<Tuple2<Long, VD>, Tuple4<Long, VD, Boolean, Boolean>>) v -> {
                GraphVertex<VD> gv = new GraphVertex<>(v._1(), v._2());
                return new Tuple4<>(v._1(), v._2(), fFn.call(gv), tFn.call(gv));
              },
              Encoders.tuple(
                  Encoders.LONG(),
                  graph.vertexAttrEncoder(),
                  Encoders.BOOLEAN(),
                  Encoders.BOOLEAN()))
          .toDF(GraphFrame.ID(), ATTR_COL, fromCol, toCol);

      JavaGraphDataset<VD, ED> edgeFiltered =
          edgeFilter != null ? graph.filterEdges(edgeFilter) : graph;
      Dataset<Row> eDF = edgeFiltered.edges()
          .toDF(GraphFrame.SRC(), GraphFrame.DST(), ATTR_COL);

      Dataset<Row> resDF = GraphFrame.apply(vDF, eDF).bfs()
          .fromExpr(functions.col(fromCol))
          .toExpr(functions.col(toCol))
          .maxPathLength(maxPathLength)
          .run();

      List<String> cols = Arrays.asList(resDF.columns());
      if (!cols.contains("from") || !cols.contains("to")) {
        return graph.spark().emptyDataset(SimplePath.encoder());
      }

      List<String> edgeCols = cols.stream()
          .filter(c -> c.startsWith("e") && c.length() > 1 && c.substring(1).matches("\\d+"))
          .collect(Collectors.toList());
      if (edgeCols.isEmpty()) {
        return resDF
            .select(
                functions.array(functions.col("from.id")).alias("path"),
                functions.lit(0).alias("length"))
            .as(SimplePath.encoder());
      }
      Column[] vIdCols = cols.stream()
          .filter(c -> c.equals("from") || c.equals("to")
              || (c.startsWith("v") && c.length() > 1 && c.substring(1).matches("\\d+")))
          .map(c -> functions.col(c + ".id"))
          .toArray(Column[]::new);
      return resDF
          .select(
              functions.array(vIdCols).alias("path"),
              functions.lit(edgeCols.size()).alias("length"))
          .as(SimplePath.encoder());
    }
  }

  public static final class JavaAllPaths<VD, ED> implements Serializable {
    private final JavaGraphDataset<VD, ED> graph;
    private FilterFunction<GraphVertex<VD>> fromPred;
    private FilterFunction<GraphVertex<VD>> toPred;
    private FilterFunction<GraphEdge<ED>> edgeFilter;
    private int maxPathLength = 5;
    private boolean isDirected = true;
    private boolean useLocalCheckpoints = false;
    private int checkpointInterval = 2;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaAllPaths(JavaGraphDataset<VD, ED> graph) {
      this.graph = graph;
    }

    public JavaAllPaths<VD, ED> fromPred(FilterFunction<GraphVertex<VD>> predicate) {
      this.fromPred = predicate;
      return this;
    }

    public JavaAllPaths<VD, ED> toPred(FilterFunction<GraphVertex<VD>> predicate) {
      this.toPred = predicate;
      return this;
    }

    public JavaAllPaths<VD, ED> edgeFilter(FilterFunction<GraphEdge<ED>> predicate) {
      this.edgeFilter = predicate;
      return this;
    }

    public JavaAllPaths<VD, ED> maxPathLength(int value) {
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
      this.storageLevel = value;
      return this;
    }

    public Dataset<SimplePath> run() {
      Objects.requireNonNull(fromPred, "fromPred is required");
      Objects.requireNonNull(toPred, "toPred is required");
      FilterFunction<GraphVertex<VD>> fFn = fromPred;
      FilterFunction<GraphVertex<VD>> tFn = toPred;

      String fromCol = "_gf_jap_from";
      String toCol = "_gf_jap_to";
      Dataset<Row> vDF = graph.vertices()
          .map(
              (MapFunction<Tuple2<Long, VD>, Tuple3<Long, Boolean, Boolean>>) v -> {
                GraphVertex<VD> gv = new GraphVertex<>(v._1(), v._2());
                return new Tuple3<>(v._1(), fFn.call(gv), tFn.call(gv));
              },
              Encoders.tuple(Encoders.LONG(), Encoders.BOOLEAN(), Encoders.BOOLEAN()))
          .toDF(GraphFrame.ID(), fromCol, toCol);

      JavaGraphDataset<VD, ED> edgeFiltered =
          edgeFilter != null ? graph.filterEdges(edgeFilter) : graph;
      Dataset<Row> eDF = edgeFiltered.edges()
          .toDF(GraphFrame.SRC(), GraphFrame.DST(), ATTR_COL)
          .select(GraphFrame.SRC(), GraphFrame.DST());

      org.graphframes.lib.AllPaths ap = GraphFrame.apply(vDF, eDF).allPaths();
      ap.fromExpr(functions.col(fromCol));
      ap.toExpr(functions.col(toCol));
      ap.maxPathLength(maxPathLength);
      ap.setIsDirected(isDirected);
      ap.setUseLocalCheckpoints(useLocalCheckpoints);
      ap.setCheckpointInterval(checkpointInterval);
      ap.setIntermediateStorageLevel(storageLevel);
      return ap.run()
          .select(functions.col("path"), functions.col("len").cast("int").alias("length"))
          .as(SimplePath.encoder());
    }
  }

  public static final class JavaSVDPlusPlus<VD, ED> implements Serializable {
    private final JavaGraphDataset<VD, ED> graph;
    private int rank = 10;
    private Integer maxIter = null;
    private double minVal = 0.0;
    private double maxVal = 5.0;
    private double gamma1 = 0.007;
    private double gamma2 = 0.007;
    private double gamma6 = 0.005;
    private double gamma7 = 0.015;

    JavaSVDPlusPlus(JavaGraphDataset<VD, ED> graph) {
      this.graph = graph;
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

      Dataset<SVDPlusPlusResult> typed = svd.run()
          .select(
              functions.col(GraphFrame.ID()).alias("id"),
              functions.col(SVDPlusPlus.COLUMN1()).alias("column1"),
              functions.col(SVDPlusPlus.COLUMN2()).alias("column2"),
              functions.col(SVDPlusPlus.COLUMN3()).alias("column3"),
              functions.col(SVDPlusPlus.COLUMN4()).alias("column4"))
          .as(SVDPlusPlusResult.encoder());
      return new Tuple2<>(typed, svd.loss());
    }
  }
}
