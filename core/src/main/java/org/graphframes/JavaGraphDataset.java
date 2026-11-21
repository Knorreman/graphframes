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
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
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

  private static final String ID_COL = "id";
  private static final String ATTR_COL = "attr";
  private static final String SRC_ID_COL = "srcId";
  private static final String DST_ID_COL = "dstId";
  private static final String SRC_ATTR_COL = "srcAttr";
  private static final String DST_ATTR_COL = "dstAttr";
  private static final String KEY_COL = "k";
  private static final String CNT_COL = "cnt";
  private static final String CLUSTER_COL = "cluster";
  private static final String PIC_WEIGHT_COL = "_gf_pic_weight";
  private static final String COMPONENT_COL = "component";
  private static final String CONNECTED_COMPONENT_COL = "connectedComponent";
  private static final String PAGERANK_COL = "pagerank";
  private static final String WEIGHT_COL = "weight";
  private static final String DISTANCES_COL = "distances";
  private static final String COUNT_COL = "count";
  private static final String LABEL_COL = "label";
  private static final String KCORE_COL = "kcore";
  private static final String FROM_COL = "from";
  private static final String TO_COL = "to";
  private static final String PATH_COL = "path";
  private static final String LEN_COL = "len";
  private static final String LENGTH_COL = "length";
  private static final String BFS_FROM_FLAG = "_gf_jbfs_from";
  private static final String BFS_TO_FLAG = "_gf_jbfs_to";
  private static final String ALL_PATHS_FROM_FLAG = "_gf_jap_from";
  private static final String ALL_PATHS_TO_FLAG = "_gf_jap_to";
  private static final String COLUMN1_COL = "column1";
  private static final String COLUMN2_COL = "column2";
  private static final String COLUMN3_COL = "column3";
  private static final String COLUMN4_COL = "column4";
  private static final String TUPLE_1 = "_1";
  private static final String TUPLE_2 = "_2";
  private static final String TUPLE_3 = "_3";
  private static final String TUPLE_4 = "_4";
  private static final String TUPLE_5 = "_5";

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
    Objects.requireNonNull(edges, "edges must not be null");
    Objects.requireNonNull(vdEncoder, "vdEncoder must not be null");
    Objects.requireNonNull(edEncoder, "edEncoder must not be null");
    Dataset<Long> ids = edges
        .toDF(SRC_ID_COL, DST_ID_COL, ATTR_COL)
        .select(functions.explode(
                functions.array(functions.col(SRC_ID_COL), functions.col(DST_ID_COL)))
            .alias(ID_COL))
        .distinct()
        .as(Encoders.LONG());
    Dataset<Tuple2<Long, VD>> vertices = ids.map(
        (MapFunction<Long, Tuple2<Long, VD>>) id -> new Tuple2<>(id, defaultVertexAttr),
        Encoders.tuple(Encoders.LONG(), vdEncoder));
    return new JavaGraphDataset<>(vertices, edges, vdEncoder, edEncoder);
  }

  private static boolean isIntegralType(DataType dt) {
    return DataTypes.ByteType.equals(dt)
        || DataTypes.ShortType.equals(dt)
        || DataTypes.IntegerType.equals(dt)
        || DataTypes.LongType.equals(dt);
  }

  /**
   * Constructs a {@link JavaGraphDataset} from an existing {@link GraphFrame} whose vertices have
   * an integral {@code id} column ({@code Byte}, {@code Short}, {@code Int}, or {@code Long}) and
   * an {@code attr} column, and whose edges have integral {@code src} and {@code dst} columns and
   * an {@code attr} column.
   */
  public static <VD, ED> JavaGraphDataset<VD, ED> fromGraphFrame(
      GraphFrame graph,
      Encoder<VD> vdEncoder,
      Encoder<ED> edEncoder) {
    Objects.requireNonNull(graph, "graph must not be null");
    Objects.requireNonNull(vdEncoder, "vdEncoder must not be null");
    Objects.requireNonNull(edEncoder, "edEncoder must not be null");
    DataType idType = graph.vertices().schema().apply(GraphFrame.ID()).dataType();
    if (!isIntegralType(idType)) {
      throw new IllegalArgumentException(
          "GraphDataset vertex ID column '" + GraphFrame.ID()
              + "' must be an integral type (Byte, Short, Int, Long), but found " + idType + ".");
    }
    DataType srcType = graph.edges().schema().apply(GraphFrame.SRC()).dataType();
    if (!isIntegralType(srcType)) {
      throw new IllegalArgumentException(
          "GraphDataset edge column '" + GraphFrame.SRC()
              + "' must be an integral type (Byte, Short, Int, Long), but found " + srcType + ".");
    }
    DataType dstType = graph.edges().schema().apply(GraphFrame.DST()).dataType();
    if (!isIntegralType(dstType)) {
      throw new IllegalArgumentException(
          "GraphDataset edge column '" + GraphFrame.DST()
              + "' must be an integral type (Byte, Short, Int, Long), but found " + dstType + ".");
    }
    List<String> vCols = Arrays.asList(graph.vertices().columns());
    if (!vCols.contains(ATTR_COL)) {
      throw new IllegalArgumentException(
          "Vertex attribute column '" + ATTR_COL + "' is missing from GraphFrame.vertices (columns: "
              + String.join(", ", vCols) + ").");
    }
    List<String> eCols = Arrays.asList(graph.edges().columns());
    if (!eCols.contains(ATTR_COL)) {
      throw new IllegalArgumentException(
          "Edge attribute column '" + ATTR_COL + "' is missing from GraphFrame.edges (columns: "
              + String.join(", ", eCols) + ").");
    }

    Encoder<Tuple2<Long, VD>> vEnc = Encoders.tuple(Encoders.LONG(), vdEncoder);
    Encoder<Tuple3<Long, Long, ED>> eEnc =
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), edEncoder);
    Dataset<Tuple2<Long, VD>> vv = graph.vertices()
        .select(
            functions.col(GraphFrame.ID()).cast("long").alias(TUPLE_1),
            functions.col(ATTR_COL).alias(TUPLE_2))
        .as(vEnc);
    Dataset<Tuple3<Long, Long, ED>> ee = graph.edges()
        .select(
            functions.col(GraphFrame.SRC()).cast("long").alias(TUPLE_1),
            functions.col(GraphFrame.DST()).cast("long").alias(TUPLE_2),
            functions.col(ATTR_COL).alias(TUPLE_3))
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
    Dataset<Row> srcV = vertices().toDF(SRC_ID_COL, SRC_ATTR_COL);
    Dataset<Row> dstV = vertices().toDF(DST_ID_COL, DST_ATTR_COL);
    return edges()
        .toDF(SRC_ID_COL, DST_ID_COL, ATTR_COL)
        .join(srcV, SRC_ID_COL)
        .join(dstV, DST_ID_COL)
        .select(
            functions.col(SRC_ID_COL).alias(TUPLE_1),
            functions.col(DST_ID_COL).alias(TUPLE_2),
            functions.col(SRC_ATTR_COL).alias(TUPLE_3),
            functions.col(DST_ATTR_COL).alias(TUPLE_4),
            functions.col(ATTR_COL).alias(TUPLE_5))
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
    validate(true, StorageLevel.MEMORY_AND_DISK());
  }

  /** Validates graph consistency with custom options without altering caller caches. */
  public void validate(boolean checkVertices, StorageLevel intermediateStorageLevel) {
    Objects.requireNonNull(
        intermediateStorageLevel, "intermediateStorageLevel must not be null");
    Dataset<Row> vIds = vertices().toDF(GraphFrame.ID(), ATTR_COL).select(GraphFrame.ID());
    Dataset<Row> eIds = edges()
        .toDF(GraphFrame.SRC(), GraphFrame.DST(), ATTR_COL)
        .select(GraphFrame.SRC(), GraphFrame.DST());
    GraphFrame.apply(vIds, eIds).validate(checkVertices, intermediateStorageLevel);
  }

  /** Returns a new graph with all edge directions reversed. */
  public JavaGraphDataset<VD, ED> asReversed() {
    Dataset<Tuple3<Long, Long, ED>> rev = edges()
        .toDF(TUPLE_1, TUPLE_2, TUPLE_3)
        .select(
            functions.col(TUPLE_2).alias(TUPLE_1),
            functions.col(TUPLE_1).alias(TUPLE_2),
            functions.col(TUPLE_3))
        .as(edgeEncoder());
    return new JavaGraphDataset<>(vertices(), rev, vdEncoder, edEncoder);
  }

  /**
   * Returns a new undirected graph adding a reverse edge for every directed edge.
   *
   * <p>Note: self-loops ({@code srcId == dstId}) are duplicated in the resulting edge dataset.
   */
  public JavaGraphDataset<VD, ED> asUndirected() {
    Dataset<Tuple3<Long, Long, ED>> rev = edges()
        .toDF(TUPLE_1, TUPLE_2, TUPLE_3)
        .select(
            functions.col(TUPLE_2).alias(TUPLE_1),
            functions.col(TUPLE_1).alias(TUPLE_2),
            functions.col(TUPLE_3))
        .as(edgeEncoder());
    return new JavaGraphDataset<>(vertices(), edges().union(rev), vdEncoder, edEncoder);
  }

  // ============== Degree metrics ==============

  public Dataset<Tuple2<Long, Integer>> outDegrees() {
    return edges()
        .toDF(SRC_ID_COL, DST_ID_COL, ATTR_COL)
        .groupBy(functions.col(SRC_ID_COL))
        .agg(functions.count("*").cast("int"))
        .toDF(TUPLE_1, TUPLE_2)
        .as(Encoders.tuple(Encoders.LONG(), Encoders.INT()));
  }

  public Dataset<Tuple2<Long, Integer>> inDegrees() {
    return edges()
        .toDF(SRC_ID_COL, DST_ID_COL, ATTR_COL)
        .groupBy(functions.col(DST_ID_COL))
        .agg(functions.count("*").cast("int"))
        .toDF(TUPLE_1, TUPLE_2)
        .as(Encoders.tuple(Encoders.LONG(), Encoders.INT()));
  }

  public Dataset<Tuple2<Long, Integer>> degrees() {
    return edges()
        .toDF(SRC_ID_COL, DST_ID_COL, ATTR_COL)
        .select(functions.explode(
                functions.array(functions.col(SRC_ID_COL), functions.col(DST_ID_COL)))
            .alias(ID_COL))
        .groupBy(functions.col(ID_COL))
        .agg(functions.count("*").cast("int"))
        .toDF(TUPLE_1, TUPLE_2)
        .as(Encoders.tuple(Encoders.LONG(), Encoders.INT()));
  }

  /**
   * Computes per-edge-type out-degrees as {@code Dataset<Tuple3<Long, K, Integer>>}
   * {@code (vertexId, edgeType, count)}. {@code null} edge categories are ignored.
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeOutDegree(
      MapFunction<ED, K> edgeTypeExtractor, Encoder<K> keyEncoder) {
    Objects.requireNonNull(edgeTypeExtractor, "edgeTypeExtractor must not be null");
    Objects.requireNonNull(keyEncoder, "keyEncoder must not be null");
    Dataset<Tuple2<Long, K>> pairs = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>)
            e -> new Tuple2<>(e._1(), edgeTypeExtractor.call(e._3())),
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, keyEncoder);
  }

  /**
   * Computes per-edge-type in-degrees as {@code Dataset<Tuple3<Long, K, Integer>>}
   * {@code (vertexId, edgeType, count)}. {@code null} edge categories are ignored.
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeInDegree(
      MapFunction<ED, K> edgeTypeExtractor, Encoder<K> keyEncoder) {
    Objects.requireNonNull(edgeTypeExtractor, "edgeTypeExtractor must not be null");
    Objects.requireNonNull(keyEncoder, "keyEncoder must not be null");
    Dataset<Tuple2<Long, K>> pairs = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>)
            e -> new Tuple2<>(e._2(), edgeTypeExtractor.call(e._3())),
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, keyEncoder);
  }

  /**
   * Computes per-edge-type total degrees as {@code Dataset<Tuple3<Long, K, Integer>>}
   * {@code (vertexId, edgeType, count)}. {@code null} edge categories are ignored.
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeDegree(
      MapFunction<ED, K> edgeTypeExtractor, Encoder<K> keyEncoder) {
    Objects.requireNonNull(edgeTypeExtractor, "edgeTypeExtractor must not be null");
    Objects.requireNonNull(keyEncoder, "keyEncoder must not be null");
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
        .toDF(ID_COL, KEY_COL)
        .filter(functions.col(KEY_COL).isNotNull())
        .groupBy(functions.col(ID_COL), functions.col(KEY_COL))
        .agg(functions.count("*").cast("int").alias(CNT_COL))
        .select(
            functions.col(ID_COL).alias(TUPLE_1),
            functions.col(KEY_COL).alias(TUPLE_2),
            functions.col(CNT_COL).alias(TUPLE_3))
        .as(Encoders.tuple(Encoders.LONG(), keyEncoder, Encoders.INT()));
  }

  // ============== Subgraphs & transformations ==============

  /** Filters vertices by predicate and removes edges referencing dropped vertices. */
  public JavaGraphDataset<VD, ED> filterVertices(FilterFunction<GraphVertex<VD>> condition) {
    Objects.requireNonNull(condition, "condition must not be null");
    Dataset<Tuple2<Long, VD>> vv = vertices().filter(
        (FilterFunction<Tuple2<Long, VD>>) v -> condition.call(new GraphVertex<>(v._1(), v._2())));
    Dataset<Row> vvIds = vv.toDF(ID_COL, ATTR_COL).select(functions.col(ID_COL));
    Dataset<Tuple3<Long, Long, ED>> ee = edges()
        .toDF(TUPLE_1, TUPLE_2, TUPLE_3)
        .join(vvIds, functions.col(TUPLE_1).equalTo(vvIds.col(ID_COL)), "left_semi")
        .join(vvIds, functions.col(TUPLE_2).equalTo(vvIds.col(ID_COL)), "left_semi")
        .as(edgeEncoder());
    return new JavaGraphDataset<>(vv, ee, vdEncoder, edEncoder);
  }

  /** Filters edges by predicate while keeping all vertices. */
  public JavaGraphDataset<VD, ED> filterEdges(FilterFunction<GraphEdge<ED>> condition) {
    Objects.requireNonNull(condition, "condition must not be null");
    Dataset<Tuple3<Long, Long, ED>> ee = edges().filter(
        (FilterFunction<Tuple3<Long, Long, ED>>)
            e -> condition.call(new GraphEdge<>(e._1(), e._2(), e._3())));
    return new JavaGraphDataset<>(vertices(), ee, vdEncoder, edEncoder);
  }

  /** Restricts the graph to vertices matching {@code vpred} and edges matching {@code epred}. */
  public JavaGraphDataset<VD, ED> subgraph(
      FilterFunction<GraphEdge<ED>> epred, FilterFunction<GraphVertex<VD>> vpred) {
    return filterEdges(epred).filterVertices(vpred);
  }

  /** Drops vertices that are not incident to any edge. */
  public JavaGraphDataset<VD, ED> dropIsolatedVertices() {
    Dataset<Row> activeIds = edges()
        .toDF(SRC_ID_COL, DST_ID_COL, ATTR_COL)
        .select(functions.explode(
                functions.array(functions.col(SRC_ID_COL), functions.col(DST_ID_COL)))
            .alias(ID_COL));
    Dataset<Tuple2<Long, VD>> vv = vertices()
        .toDF(TUPLE_1, TUPLE_2)
        .join(activeIds, functions.col(TUPLE_1).equalTo(activeIds.col(ID_COL)), "left_semi")
        .as(vertexEncoder());
    return new JavaGraphDataset<>(vv, edges(), vdEncoder, edEncoder);
  }

  /** Transforms each vertex attribute. */
  public <VD2> JavaGraphDataset<VD2, ED> mapVertices(
      Function2<Long, VD, VD2> fn, Encoder<VD2> vd2Encoder) {
    Objects.requireNonNull(fn, "fn must not be null");
    Objects.requireNonNull(vd2Encoder, "vd2Encoder must not be null");
    Dataset<Tuple2<Long, VD2>> vv = vertices().map(
        (MapFunction<Tuple2<Long, VD>, Tuple2<Long, VD2>>)
            v -> new Tuple2<>(v._1(), fn.call(v._1(), v._2())),
        Encoders.tuple(Encoders.LONG(), vd2Encoder));
    return new JavaGraphDataset<>(vv, edges(), vd2Encoder, edEncoder);
  }

  /** Transforms each edge attribute. */
  public <ED2> JavaGraphDataset<VD, ED2> mapEdges(
      MapFunction<GraphEdge<ED>, ED2> fn, Encoder<ED2> ed2Encoder) {
    Objects.requireNonNull(fn, "fn must not be null");
    Objects.requireNonNull(ed2Encoder, "ed2Encoder must not be null");
    Dataset<Tuple3<Long, Long, ED2>> ee = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple3<Long, Long, ED2>>)
            e -> new Tuple3<>(e._1(), e._2(), fn.call(new GraphEdge<>(e._1(), e._2(), e._3()))),
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), ed2Encoder));
    return new JavaGraphDataset<>(vertices(), ee, vdEncoder, ed2Encoder);
  }

  /**
   * Transforms each edge attribute using the full triplet.
   *
   * <p>Note: because this operates over {@code triplets()}, edges whose source or destination
   * vertex is absent from {@code vertices()} are omitted.
   */
  public <ED2> JavaGraphDataset<VD, ED2> mapTriplets(
      MapFunction<GraphTriplet<VD, ED>, ED2> fn, Encoder<ED2> ed2Encoder) {
    Objects.requireNonNull(fn, "fn must not be null");
    Objects.requireNonNull(ed2Encoder, "ed2Encoder must not be null");
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
            functions.col(GraphFrame.ID()).alias(ID_COL),
            functions.col(CLUSTER_COL).cast("int").alias(CLUSTER_COL))
        .as(ClusterAssignment.encoder());
  }

  /** Power Iteration Clustering with a typed edge weight extractor. */
  public Dataset<ClusterAssignment> powerIterationClustering(
      int k, int maxIter, MapFunction<ED, Double> weightExtractor) {
    Objects.requireNonNull(weightExtractor, "weightExtractor must not be null");
    Dataset<Row> vDF = vertices().toDF(GraphFrame.ID(), ATTR_COL);
    Dataset<Row> eDF = edges()
        .map(
            (MapFunction<Tuple3<Long, Long, ED>, Tuple3<Long, Long, Double>>)
                e -> new Tuple3<>(e._1(), e._2(), weightExtractor.call(e._3())),
            Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.DOUBLE()))
        .toDF(GraphFrame.SRC(), GraphFrame.DST(), PIC_WEIGHT_COL);
    return GraphFrame.apply(vDF, eDF)
        .powerIterationClustering(k, maxIter, scala.Option.apply(PIC_WEIGHT_COL))
        .select(
            functions.col(GraphFrame.ID()).alias(ID_COL),
            functions.col(CLUSTER_COL).cast("int").alias(CLUSTER_COL))
        .as(ClusterAssignment.encoder());
  }

  private static <T> Dataset<T> materializeAndReleaseRaw(
      Dataset<Row> raw, Dataset<T> typed, StorageLevel storageLevel) {
    if (!StorageLevel.NONE().equals(typed.storageLevel())) {
      typed.count();
      return typed;
    }
    Dataset<T> persisted = typed.persist(storageLevel);
    boolean succeeded = false;
    try {
      persisted.count();
      succeeded = true;
      return persisted;
    } finally {
      if (!succeeded) {
        persisted.unpersist();
      }
      if (!StorageLevel.NONE().equals(raw.storageLevel())) {
        raw.unpersist();
      }
    }
  }

  private static <VD, ED> JavaGraphDataset<VD, ED> materializeGraphAndReleaseRaw(
      GraphFrame rawGF,
      Dataset<Tuple2<Long, VD>> typedVertices,
      Dataset<Tuple3<Long, Long, ED>> typedEdges,
      Encoder<VD> vdEncoder,
      Encoder<ED> edEncoder,
      StorageLevel storageLevel) {
    boolean verticesAlreadyCached = !StorageLevel.NONE().equals(typedVertices.storageLevel());
    boolean edgesAlreadyCached = !StorageLevel.NONE().equals(typedEdges.storageLevel());
    Dataset<Tuple2<Long, VD>> persistedV =
        verticesAlreadyCached ? typedVertices : typedVertices.persist(storageLevel);
    Dataset<Tuple3<Long, Long, ED>> persistedE =
        edgesAlreadyCached ? typedEdges : typedEdges.persist(storageLevel);
    boolean succeeded = false;
    try {
      persistedV.count();
      persistedE.count();
      succeeded = true;
      return JavaGraphDataset.of(persistedV, persistedE, vdEncoder, edEncoder);
    } finally {
      if (!succeeded) {
        if (!verticesAlreadyCached) {
          persistedV.unpersist();
        }
        if (!edgesAlreadyCached) {
          persistedE.unpersist();
        }
      }
      if (!verticesAlreadyCached && !StorageLevel.NONE().equals(rawGF.vertices().storageLevel())) {
        rawGF.vertices().unpersist();
      }
      if (!edgesAlreadyCached && !StorageLevel.NONE().equals(rawGF.edges().storageLevel())) {
        rawGF.edges().unpersist();
      }
    }
  }

  // ============== Nested Algorithm Builders ==============

  /**
   * Connected Components builder. Returns a persisted and materialized dataset; callers should
   * call {@code unpersist()} when finished. Unset configuration parameters defer to Spark SQL
   * {@code spark.graphframes.*} defaults.
   */
  public static final class JavaConnectedComponents implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private String algorithm = null;
    private Integer broadcastThreshold = null;
    private Integer checkpointInterval = null;
    private Boolean useLocalCheckpoints = null;
    private StorageLevel storageLevel = null;
    private Integer maxIter = null;

    JavaConnectedComponents(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
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
      return materializeAndReleaseRaw(rawDF, typed, targetLevel);
    }
  }

  /**
   * PageRank builder. Returns a persisted and materialized {@link JavaGraphDataset}; callers
   * should call {@code unpersist()} when finished.
   */
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
      return materializeGraphAndReleaseRaw(
          resGF, vv, ee, vAttrEnc, eAttrEnc, StorageLevel.MEMORY_AND_DISK());
    }
  }

  /**
   * Shortest Paths builder. Returns a persisted and materialized dataset; callers should call
   * {@code unpersist()} when finished.
   */
  public static final class JavaShortestPaths implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private List<Long> landmarks;
    private String algorithm = "graphx";
    private boolean isDirected = true;
    private boolean useLocalCheckpoints = false;
    private int checkpointInterval = 2;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaShortestPaths(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaShortestPaths landmarks(List<Long> value) {
      this.landmarks = Objects.requireNonNull(value, "landmarks must not be null");
      return this;
    }

    /** Sets the algorithm ({@code "graphx"} or {@code "graphframes"}). */
    public JavaShortestPaths setAlgorithm(String value) {
      this.algorithm = Objects.requireNonNull(value, "algorithm must not be null");
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
      return this;
    }

    public Dataset<ShortestPathsResult> run() {
      Objects.requireNonNull(landmarks, "landmarks must be set");
      org.graphframes.lib.ShortestPaths sp = graph.toGraphFrame().shortestPaths();
      sp.landmarks(new ArrayList<>(landmarks));
      sp.setAlgorithm(algorithm);
      sp.setIsDirected(isDirected);
      sp.setUseLocalCheckpoints(useLocalCheckpoints);
      sp.setCheckpointInterval(checkpointInterval);
      sp.setIntermediateStorageLevel(storageLevel);
      Dataset<Row> rawDF = sp.run();
      Dataset<ShortestPathsResult> typed = rawDF
          .select(functions.col(GraphFrame.ID()).alias(ID_COL), functions.col(DISTANCES_COL))
          .as(ShortestPathsResult.encoder());
      return materializeAndReleaseRaw(rawDF, typed, storageLevel);
    }
  }

  /**
   * Triangle Count builder. Returns a persisted and materialized dataset; callers should call
   * {@code unpersist()} when finished.
   */
  public static final class JavaTriangleCount implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private String algorithm = "exact";
    private int lgNomEntries = 12;
    private StorageLevel storageLevel = StorageLevel.MEMORY_AND_DISK();

    JavaTriangleCount(JavaGraphDataset<?, ?> graph) {
      this.graph = graph;
    }

    public JavaTriangleCount setAlgorithm(String value) {
      this.algorithm = Objects.requireNonNull(value, "algorithm must not be null");
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
      return this;
    }

    public Dataset<TriangleCountResult> run() {
      org.graphframes.lib.TriangleCount tc = graph.toGraphFrame().triangleCount();
      tc.setAlgorithm(algorithm);
      tc.setLgNomEntries(lgNomEntries);
      tc.setIntermediateStorageLevel(storageLevel);
      Dataset<Row> rawDF = tc.run();
      Dataset<TriangleCountResult> typed = rawDF
          .select(
              functions.col(GraphFrame.ID()).alias(ID_COL),
              functions.col(COUNT_COL).cast("long").alias(COUNT_COL))
          .as(TriangleCountResult.encoder());
      return materializeAndReleaseRaw(rawDF, typed, storageLevel);
    }
  }

  /**
   * Label Propagation builder. Returns a persisted and materialized dataset; callers should call
   * {@code unpersist()} when finished.
   */
  public static final class JavaLabelPropagation implements Serializable {
    private final JavaGraphDataset<?, ?> graph;
    private Integer maxIter = null;
    private String algorithm = "graphx";
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
      return this;
    }

    public Dataset<LabelPropagationResult> run() {
      Objects.requireNonNull(maxIter, "maxIter must be set");
      org.graphframes.lib.LabelPropagation lp = graph.toGraphFrame().labelPropagation();
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
      return materializeAndReleaseRaw(rawDF, typed, storageLevel);
    }
  }

  /**
   * Strongly Connected Components builder. Returns a persisted and materialized dataset; callers
   * should call {@code unpersist()} when finished.
   */
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
      Dataset<Row> rawDF = scc.run();
      Dataset<ConnectedComponentResult> typed = rawDF
          .select(
              functions.col(GraphFrame.ID()).alias(ID_COL),
              functions.col(COMPONENT_COL).alias(CONNECTED_COMPONENT_COL))
          .as(ConnectedComponentResult.encoder());
      return materializeAndReleaseRaw(rawDF, typed, StorageLevel.MEMORY_AND_DISK());
    }
  }

  /**
   * K-Core decomposition builder. Returns a persisted and materialized dataset; callers should
   * call {@code unpersist()} when finished.
   */
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
      return this;
    }

    public Dataset<KCoreResult> run() {
      KCore kc = graph.toGraphFrame().kCore();
      kc.setUseLocalCheckpoints(useLocalCheckpoints);
      kc.setCheckpointInterval(checkpointInterval);
      kc.setIntermediateStorageLevel(storageLevel);
      Dataset<Row> rawDF = kc.run();
      Dataset<KCoreResult> typed = rawDF
          .select(
              functions.col(GraphFrame.ID()).alias(ID_COL),
              functions.col(KCore.kCoreColumnName()).alias(KCORE_COL))
          .as(KCoreResult.encoder());
      return materializeAndReleaseRaw(rawDF, typed, storageLevel);
    }
  }

  /**
   * Maximal Independent Set builder. Returns a persisted and materialized dataset; callers should
   * call {@code unpersist()} when finished.
   */
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
      return this;
    }

    public Dataset<Tuple2<Long, VD>> run(long seed) {
      org.graphframes.lib.MaximalIndependentSet mis =
          graph.toGraphFrame().maximalIndependentSet();
      mis.setUseLocalCheckpoints(useLocalCheckpoints);
      mis.setCheckpointInterval(checkpointInterval);
      mis.setIntermediateStorageLevel(storageLevel);
      Dataset<Row> rawMis = mis.run(seed);
      Dataset<Row> misIds = rawMis.select(functions.col(GraphFrame.ID()).alias(ID_COL));
      Dataset<Tuple2<Long, VD>> typed = graph.vertices()
          .toDF(TUPLE_1, TUPLE_2)
          .join(misIds, functions.col(TUPLE_1).equalTo(misIds.col(ID_COL)), "left_semi")
          .as(graph.vertexEncoder());
      return materializeAndReleaseRaw(rawMis, typed, storageLevel);
    }
  }

  /**
   * Cycle Detection builder. Returns a persisted and materialized dataset; callers should call
   * {@code unpersist()} when finished.
   */
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
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
      return materializeAndReleaseRaw(rawDF, typed, storageLevel);
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
      this.storageLevel = Objects.requireNonNull(
          value, "intermediateStorageLevel must not be null");
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

      org.graphframes.lib.AllPaths ap = GraphFrame.apply(vDF, eDF).allPaths();
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

  /**
   * SVD++ collaborative filtering builder. Returns a persisted and materialized dataset alongside
   * training loss; callers should call {@code unpersist()} when finished.
   */
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
      Dataset<SVDPlusPlusResult> persisted =
          materializeAndReleaseRaw(rawDF, typed, StorageLevel.MEMORY_AND_DISK());
      return new Tuple2<>(persisted, svd.loss());
    }
  }
}
