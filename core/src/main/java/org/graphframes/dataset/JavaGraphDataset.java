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

package org.graphframes.dataset;

import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.lib.ClusterAssignment;
import org.graphframes.dataset.lib.JavaAggregateMessages;
import org.graphframes.dataset.lib.JavaAllPaths;
import org.graphframes.dataset.lib.JavaBFS;
import org.graphframes.dataset.lib.JavaConnectedComponents;
import org.graphframes.dataset.lib.JavaDetectingCycles;
import org.graphframes.dataset.lib.JavaKCore;
import org.graphframes.dataset.lib.JavaLabelPropagation;
import org.graphframes.dataset.lib.JavaMaximalIndependentSet;
import org.graphframes.dataset.lib.JavaPageRank;
import org.graphframes.dataset.lib.JavaPregel;
import org.graphframes.dataset.lib.JavaSVDPlusPlus;
import org.graphframes.dataset.lib.JavaShortestPaths;
import org.graphframes.dataset.lib.JavaStronglyConnectedComponents;
import org.graphframes.dataset.lib.JavaTriangleCount;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple5;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

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
   * Computes per-edge-type out-degrees restricted to categories in {@code edgeTypes}.
   *
   * <p>{@code null} entries in {@code edgeTypes} and edges whose category is {@code null} or not
   * in {@code edgeTypes} are ignored. Unlike Scala {@code GraphDataset.typeOutDegree}, returns
   * sparse {@code (vertexId, edgeType, count)} rows (without zero-filled entries for absent edge
   * types).
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeOutDegree(
      MapFunction<ED, K> edgeTypeExtractor, List<K> edgeTypes, Encoder<K> keyEncoder) {
    Objects.requireNonNull(edgeTypeExtractor, "edgeTypeExtractor must not be null");
    Objects.requireNonNull(edgeTypes, "edgeTypes must not be null");
    Objects.requireNonNull(keyEncoder, "keyEncoder must not be null");
    Dataset<Tuple2<Long, K>> pairs = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>)
            e -> new Tuple2<>(e._1(), edgeTypeExtractor.call(e._3())),
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, edgeTypes, keyEncoder);
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
   * Computes per-edge-type in-degrees restricted to categories in {@code edgeTypes}.
   *
   * <p>{@code null} entries in {@code edgeTypes} and edges whose category is {@code null} or not
   * in {@code edgeTypes} are ignored. Unlike Scala {@code GraphDataset.typeInDegree}, returns
   * sparse {@code (vertexId, edgeType, count)} rows (without zero-filled entries for absent edge
   * types).
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeInDegree(
      MapFunction<ED, K> edgeTypeExtractor, List<K> edgeTypes, Encoder<K> keyEncoder) {
    Objects.requireNonNull(edgeTypeExtractor, "edgeTypeExtractor must not be null");
    Objects.requireNonNull(edgeTypes, "edgeTypes must not be null");
    Objects.requireNonNull(keyEncoder, "keyEncoder must not be null");
    Dataset<Tuple2<Long, K>> pairs = edges().map(
        (MapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>)
            e -> new Tuple2<>(e._2(), edgeTypeExtractor.call(e._3())),
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, edgeTypes, keyEncoder);
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

  /**
   * Computes per-edge-type total degrees restricted to categories in {@code edgeTypes}.
   *
   * <p>{@code null} entries in {@code edgeTypes} and edges whose category is {@code null} or not
   * in {@code edgeTypes} are ignored. Unlike Scala {@code GraphDataset.typeDegree}, returns sparse
   * {@code (vertexId, edgeType, count)} rows (without zero-filled entries for absent edge types).
   */
  public <K> Dataset<Tuple3<Long, K, Integer>> typeDegree(
      MapFunction<ED, K> edgeTypeExtractor, List<K> edgeTypes, Encoder<K> keyEncoder) {
    Objects.requireNonNull(edgeTypeExtractor, "edgeTypeExtractor must not be null");
    Objects.requireNonNull(edgeTypes, "edgeTypes must not be null");
    Objects.requireNonNull(keyEncoder, "keyEncoder must not be null");
    Dataset<Tuple2<Long, K>> pairs = edges().flatMap(
        (FlatMapFunction<Tuple3<Long, Long, ED>, Tuple2<Long, K>>) e -> {
          K k = edgeTypeExtractor.call(e._3());
          return Arrays.asList(new Tuple2<>(e._1(), k), new Tuple2<>(e._2(), k)).iterator();
        },
        Encoders.tuple(Encoders.LONG(), keyEncoder));
    return countByKey(pairs, edgeTypes, keyEncoder);
  }

  private <K> Dataset<Tuple3<Long, K, Integer>> countByKey(
      Dataset<Tuple2<Long, K>> pairs, List<K> edgeTypes, Encoder<K> keyEncoder) {
    HashSet<K> allowed = new HashSet<>();
    for (K k : edgeTypes) {
      if (k != null) {
        allowed.add(k);
      }
    }
    if (allowed.isEmpty()) {
      return spark().emptyDataset(Encoders.tuple(Encoders.LONG(), keyEncoder, Encoders.INT()));
    }
    Dataset<Tuple2<Long, K>> filtered = pairs.filter(
        (FilterFunction<Tuple2<Long, K>>) p -> p._2() != null && allowed.contains(p._2()));
    return countByKey(filtered, keyEncoder);
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
}
