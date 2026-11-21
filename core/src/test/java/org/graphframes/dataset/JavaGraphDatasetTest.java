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

import org.apache.spark.graphframes.graphx.EdgeDirection;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Aggregator;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.lib.ClusterAssignment;
import org.graphframes.dataset.lib.ConnectedComponentResult;
import org.graphframes.dataset.lib.KCoreResult;
import org.graphframes.dataset.lib.LabelPropagationResult;
import org.graphframes.dataset.lib.SVDPlusPlusResult;
import org.graphframes.dataset.lib.ShortestPathsResult;
import org.graphframes.dataset.lib.SimplePath;
import org.graphframes.dataset.lib.TriangleCountResult;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple5;

import java.io.Serializable;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Pure-Java test suite verifying {@link JavaGraphDataset}, {@code JavaAggregateMessages},
 * {@code JavaPregel}, and all Java algorithm wrappers with JavaBean and primitive encoders.
 */
public class JavaGraphDatasetTest {

  public static class PersonBean implements Serializable {
    private String name;
    private int age;

    public PersonBean() {}

    public PersonBean(String name, int age) {
      this.name = name;
      this.age = age;
    }

    public String getName() {
      return name;
    }

    public void setName(String name) {
      this.name = name;
    }

    public int getAge() {
      return age;
    }

    public void setAge(int age) {
      this.age = age;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof PersonBean)) return false;
      PersonBean that = (PersonBean) o;
      return age == that.age && Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
      return Objects.hash(name, age);
    }
  }

  public static class EdgeBean implements Serializable {
    private String kind;
    private double weight;

    public EdgeBean() {}

    public EdgeBean(String kind, double weight) {
      this.kind = kind;
      this.weight = weight;
    }

    public String getKind() {
      return kind;
    }

    public void setKind(String kind) {
      this.kind = kind;
    }

    public double getWeight() {
      return weight;
    }

    public void setWeight(double weight) {
      this.weight = weight;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof EdgeBean)) return false;
      EdgeBean that = (EdgeBean) o;
      return Double.compare(that.weight, weight) == 0 && Objects.equals(kind, that.kind);
    }

    @Override
    public int hashCode() {
      return Objects.hash(kind, weight);
    }
  }

  private static void assertEq(Object expected, Object actual, String msg) {
    if (!Objects.equals(expected, actual)) {
      throw new AssertionError(msg + ": expected <" + expected + "> but got <" + actual + ">");
    }
  }

  private static JavaGraphDataset<PersonBean, EdgeBean> sampleGraph(SparkSession spark) {
    Dataset<Tuple2<Long, PersonBean>> vv = spark.createDataset(
        Arrays.asList(
            new Tuple2<>(1L, new PersonBean("Alice", 30)),
            new Tuple2<>(2L, new PersonBean("Bob", 25)),
            new Tuple2<>(3L, new PersonBean("Charlie", 40)),
            new Tuple2<>(4L, new PersonBean("Isolated", 50))),
        Encoders.tuple(Encoders.LONG(), Encoders.bean(PersonBean.class)));
    Dataset<Tuple3<Long, Long, EdgeBean>> ee = spark.createDataset(
        Arrays.asList(
            new Tuple3<>(1L, 2L, new EdgeBean("friend", 1.0)),
            new Tuple3<>(2L, 3L, new EdgeBean("follow", 2.0)),
            new Tuple3<>(3L, 1L, new EdgeBean("friend", 3.0))),
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.bean(EdgeBean.class)));
    return JavaGraphDataset.of(
        vv, ee, Encoders.bean(PersonBean.class), Encoders.bean(EdgeBean.class));
  }

  public static void testCoreOperationsAndJavaBeans(SparkSession spark) {
    JavaGraphDataset<PersonBean, EdgeBean> g = sampleGraph(spark);
    g.validate();

    // Round-trip via GraphFrame preserves JavaBeans
    JavaGraphDataset<PersonBean, EdgeBean> rt = JavaGraphDataset.fromGraphFrame(
        g.toGraphFrame(), Encoders.bean(PersonBean.class), Encoders.bean(EdgeBean.class));
    assertEq(new HashSet<>(g.vertices().collectAsList()),
        new HashSet<>(rt.vertices().collectAsList()), "round-trip vertices");
    assertEq(new HashSet<>(g.edges().collectAsList()),
        new HashSet<>(rt.edges().collectAsList()), "round-trip edges");

    // fromEdges
    JavaGraphDataset<String, EdgeBean> fromEdgesGraph = JavaGraphDataset.fromEdges(
        g.edges(), "default", Encoders.STRING(), Encoders.bean(EdgeBean.class));
    assertEq(3L, fromEdgesGraph.vertices().count(), "fromEdges vertex count");

    // Triplets
    List<Tuple5<Long, Long, PersonBean, PersonBean, EdgeBean>> triplets =
        g.triplets().collectAsList();
    assertEq(3, triplets.size(), "triplet count");

    // Degrees & typed degrees
    Map<Long, Integer> degMap = new HashMap<>();
    for (Tuple2<Long, Integer> t : g.degrees().collectAsList()) {
      degMap.put(t._1(), t._2());
    }
    assertEq(2, degMap.get(1L), "degree of 1");
    assertEq(2, degMap.get(2L), "degree of 2");
    assertEq(2, degMap.get(3L), "degree of 3");

    List<Tuple3<Long, String, Integer>> outByKind =
        g.typeOutDegree(EdgeBean::getKind, Encoders.STRING()).collectAsList();
    assertEq(3, outByKind.size(), "typeOutDegree rows");

    // filterVertices, filterEdges, dropIsolatedVertices
    JavaGraphDataset<PersonBean, EdgeBean> dropped = g.dropIsolatedVertices();
    assertEq(3L, dropped.vertices().count(), "dropIsolatedVertices");

    JavaGraphDataset<PersonBean, EdgeBean> filtered =
        g.filterVertices(v -> v.attr().getAge() <= 35);
    assertEq(2L, filtered.vertices().count(), "filterVertices vertices");
    assertEq(1L, filtered.edges().count(), "filterVertices edges");

    // mapVertices, mapEdges, mapTriplets, asReversed, asUndirected
    JavaGraphDataset<String, Double> mapped = g
        .mapVertices((id, p) -> p.getName(), Encoders.STRING())
        .mapEdges(e -> e.attr().getWeight(), Encoders.DOUBLE());
    assertEq(4L, mapped.vertices().count(), "mapVertices");
    assertEq(3L, mapped.edges().count(), "mapEdges");
    assertEq(6L, g.asUndirected().edges().count(), "asUndirected");
    assertEq(3L, g.asReversed().edges().count(), "asReversed");
  }

  public static class SumAggregator extends Aggregator<Integer, Long, Long> {
    @Override
    public Long zero() {
      return 0L;
    }

    @Override
    public Long reduce(Long b, Integer a) {
      return b + a;
    }

    @Override
    public Long merge(Long b1, Long b2) {
      return b1 + b2;
    }

    @Override
    public Long finish(Long reduction) {
      return reduction;
    }

    @Override
    public Encoder<Long> bufferEncoder() {
      return Encoders.LONG();
    }

    @Override
    public Encoder<Long> outputEncoder() {
      return Encoders.LONG();
    }
  }

  public static void testAggregateMessagesAndPregel(SparkSession spark) {
    JavaGraphDataset<PersonBean, EdgeBean> g = sampleGraph(spark);

    // AggregateMessages with sendToDst + sendToSrcOpt
    Map<Long, Integer> aggMap = new HashMap<>();
    for (Tuple2<Long, Integer> r : g.aggregateMessages(Encoders.INT())
        .sendToDst(t -> t.srcAttr().getAge())
        .sendToSrcOpt(t -> "friend".equals(t.attr().getKind())
            ? Optional.of(t.dstAttr().getAge())
            : Optional.empty())
        .agg(Integer::sum)
        .collectAsList()) {
      aggMap.put(r._1(), r._2());
    }
    // 1 receives dst(3->1)=40 + src(1->2 friend)=25 => 65
    // 2 receives dst(1->2)=30 => 30
    // 3 receives dst(2->3)=25 + src(3->1 friend)=30 => 55
    assertEq(65, aggMap.get(1L), "agg 1");
    assertEq(30, aggMap.get(2L), "agg 2");
    assertEq(55, aggMap.get(3L), "agg 3");

    // Pregel: chain propagation on 1 -> 2 -> 3 -> 4
    Dataset<Tuple2<Long, Integer>> vv = spark.createDataset(
        Arrays.asList(
            new Tuple2<>(1L, 1),
            new Tuple2<>(2L, 0),
            new Tuple2<>(3L, 0),
            new Tuple2<>(4L, 0)),
        Encoders.tuple(Encoders.LONG(), Encoders.INT()));
    Dataset<Tuple3<Long, Long, Integer>> ee = spark.createDataset(
        Arrays.asList(
            new Tuple3<>(1L, 2L, 1),
            new Tuple3<>(2L, 3L, 1),
            new Tuple3<>(3L, 4L, 1)),
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.INT()));
    JavaGraphDataset<Integer, Integer> chain =
        JavaGraphDataset.of(vv, ee, Encoders.INT(), Encoders.INT());

    JavaGraphDataset<Integer, Integer> res = chain.pregel()
        .setMaxIter(10)
        .setCheckpointInterval(2)
        .setUseLocalCheckpoints(true)
        .setActiveDirection(EdgeDirection.Out())
        .withMessages(0, Encoders.INT())
        .vprog((id, attr, msg) -> Math.max(attr, msg))
        .sendMsgToDstOpt(t -> t.srcAttr() > t.dstAttr()
            ? Optional.of(t.srcAttr())
            : Optional.empty())
        .mergeMsg(Math::max)
        .run();

    for (Tuple2<Long, Integer> v : res.vertices().collectAsList()) {
      assertEq(1, v._2(), "pregel value at " + v._1());
    }
    res.vertices().unpersist();
  }

  public static void testAlgorithmsAndTraversals(SparkSession spark) {
    JavaGraphDataset<PersonBean, EdgeBean> g = sampleGraph(spark);

    // Connected components
    Map<Long, Long> cc = new HashMap<>();
    for (ConnectedComponentResult r :
        g.connectedComponents().setUseLocalCheckpoints(true).run().collectAsList()) {
      cc.put(r.id(), r.connectedComponent());
    }
    assertEq(1L, cc.get(1L), "cc 1");
    assertEq(1L, cc.get(2L), "cc 2");
    assertEq(1L, cc.get(3L), "cc 3");
    assertEq(4L, cc.get(4L), "cc 4");

    // PageRank preserves JavaBeans
    JavaGraphDataset<Tuple2<PersonBean, Double>, Tuple2<EdgeBean, Double>> pr =
        g.pageRank().resetProbability(0.15).maxIter(3).run();
    Map<Long, Tuple2<PersonBean, Double>> prVerts = new HashMap<>();
    for (Tuple2<Long, Tuple2<PersonBean, Double>> v : pr.vertices().collectAsList()) {
      prVerts.put(v._1(), v._2());
    }
    assertEq(new PersonBean("Alice", 30), prVerts.get(1L)._1(), "pagerank attr");
    if (prVerts.get(1L)._2() <= 0.0) {
      throw new AssertionError("pagerank should be positive");
    }

    // Shortest paths
    List<ShortestPathsResult> sp = g.shortestPaths().landmarks(Arrays.asList(3L)).run()
        .collectAsList();
    assertEq(4, sp.size(), "shortestPaths count");

    // Triangle count
    Map<Long, Long> tc = new HashMap<>();
    for (TriangleCountResult r : g.triangleCount().run().collectAsList()) {
      tc.put(r.id(), r.count());
    }
    assertEq(1L, tc.get(1L), "triangle count 1");
    assertEq(0L, tc.get(4L), "triangle count 4");

    // Strongly connected components
    List<ConnectedComponentResult> scc = g.stronglyConnectedComponents().maxIter(5).run()
        .collectAsList();
    assertEq(4, scc.size(), "scc count");

    // KCore
    Map<Long, Integer> kc = new HashMap<>();
    for (KCoreResult r : g.kCore().setUseLocalCheckpoints(true).run().collectAsList()) {
      kc.put(r.id(), r.kcore());
    }
    assertEq(2, kc.get(1L), "kcore 1");
    assertEq(0, kc.get(4L), "kcore 4");

    // Label propagation
    List<LabelPropagationResult> lp = g.labelPropagation().maxIter(3).run().collectAsList();
    assertEq(4, lp.size(), "labelPropagation count");

    // Maximal independent set preserves JavaBean VD
    List<Tuple2<Long, PersonBean>> mis =
        g.maximalIndependentSet().setUseLocalCheckpoints(true).run(42L).collectAsList();
    if (mis.isEmpty()) {
      throw new AssertionError("MIS should be non-empty");
    }

    // Detecting cycles
    List<SimplePath> cycles =
        g.detectingCycles().setUseLocalCheckpoints(true).run().collectAsList();
    assertEq(1, cycles.size(), "cycle count");
    assertEq(3, cycles.get(0).length(), "cycle length");

    // BFS & AllPaths
    List<SimplePath> bfsPaths = g.bfs()
        .fromPred(v -> "Alice".equals(v.attr().getName()))
        .toPred(v -> "Charlie".equals(v.attr().getName()))
        .maxPathLength(4)
        .run()
        .collectAsList();
    assertEq(1, bfsPaths.size(), "bfs path count");
    assertEq(2, bfsPaths.get(0).length(), "bfs path length");

    List<SimplePath> allPaths = g.allPaths()
        .fromPred(v -> v.id() == 1L)
        .toPred(v -> v.id() == 3L)
        .maxPathLength(4)
        .setUseLocalCheckpoints(true)
        .run()
        .collectAsList();
    assertEq(1, allPaths.size(), "allPaths count");

    // SVD++ & PowerIterationClustering (operate on edges, so only connected vertices 1, 2, 3)
    Tuple2<Dataset<SVDPlusPlusResult>, Double> svd =
        g.svdPlusPlus().rank(4).maxIter(2).run(EdgeBean::getWeight);
    assertEq(3L, svd._1().count(), "svd++ count");

    Set<Long> picIds = g.powerIterationClustering(2, 5, EdgeBean::getWeight)
        .collectAsList()
        .stream()
        .map(ClusterAssignment::id)
        .collect(Collectors.toSet());
    assertEq(new HashSet<>(Arrays.asList(1L, 2L, 3L)), picIds, "PIC vertex IDs");
  }

  public static void testReviewRegressionFixes(SparkSession spark) {
    // 1. fromGraphFrame rejects non-integral IDs and missing attr columns
    Dataset<Row> validVertices = spark.createDataset(
        Collections.singletonList(new Tuple2<>(1L, "A")),
        Encoders.tuple(Encoders.LONG(), Encoders.STRING())).toDF("id", "attr");
    Dataset<Row> validEdges = spark.createDataset(
        Collections.singletonList(new Tuple3<>(1L, 1L, "self")),
        Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.STRING()))
        .toDF("src", "dst", "attr");
    Dataset<Row> strV = spark.createDataset(
        Collections.singletonList(new Tuple2<>("1", "A")),
        Encoders.tuple(Encoders.STRING(), Encoders.STRING())).toDF("id", "attr");
    boolean rejectedStr = false;
    try {
      JavaGraphDataset.fromGraphFrame(
          GraphFrame.apply(strV, validEdges), Encoders.STRING(), Encoders.STRING());
    } catch (IllegalArgumentException expected) {
      rejectedStr = true;
    }
    assertEq(true, rejectedStr, "fromGraphFrame string id rejection");

    Dataset<Row> noAttrV = validVertices.withColumnRenamed("attr", "other");
    boolean rejectedAttr = false;
    try {
      JavaGraphDataset.fromGraphFrame(
          GraphFrame.apply(noAttrV, validEdges), Encoders.STRING(), Encoders.STRING());
    } catch (IllegalArgumentException expected) {
      rejectedAttr = true;
    }
    assertEq(true, rejectedAttr, "fromGraphFrame missing attr rejection");

    // 2. JavaAggregator, JavaBFS 0-hop/no-path/edgeFilter/validation, JavaAllPaths validation
    JavaGraphDataset<PersonBean, EdgeBean> g = sampleGraph(spark);
    g.cache();
    g.vertices().count();
    g.edges().count();
    try {
      g.validate();
      assertEq(false, StorageLevel.NONE().equals(g.vertices().storageLevel()),
          "validate keeps cached vertices");

      Map<Long, Long> customAgg = new HashMap<>();
      for (Tuple2<Long, Long> r : g.aggregateMessages(Encoders.INT())
          .sendToDst(t -> t.srcAttr().getAge())
          .agg(new SumAggregator())
          .collectAsList()) {
        customAgg.put(r._1(), r._2());
      }
      assertEq(30L, customAgg.get(2L), "Aggregator agg for vertex 2");

      List<SimplePath> zeroHop = g.bfs()
          .fromPred(v -> v.id() == 1L)
          .toPred(v -> v.id() == 1L)
          .run()
          .collectAsList();
      assertEq(1, zeroHop.size(), "JavaBFS 0-hop count");
      assertEq(Arrays.asList(1L), zeroHop.get(0).pathAsJava(), "SimplePath.pathAsJava");

      List<SimplePath> noPath = g.bfs()
          .fromPred(v -> v.id() == 1L)
          .toPred(v -> v.id() == 4L)
          .edgeFilter(e -> "friend".equals(e.attr().getKind()))
          .maxPathLength(3)
          .run()
          .collectAsList();
      assertEq(0, noPath.size(), "JavaBFS no-path count");

      // JavaBFS & JavaAllPaths eager maxPathLength validation
      boolean caughtBfsLen = false;
      try {
        g.bfs().maxPathLength(-1);
      } catch (IllegalArgumentException expected) {
        caughtBfsLen = true;
      }
      assertEq(true, caughtBfsLen, "JavaBFS rejects maxPathLength < 0");

      boolean caughtApLen = false;
      try {
        g.allPaths().maxPathLength(0);
      } catch (IllegalArgumentException expected) {
        caughtApLen = true;
      }
      assertEq(true, caughtApLen, "JavaAllPaths rejects maxPathLength <= 0");

      // Java collection helpers on ShortestPathsResult & SVDPlusPlusResult
      Dataset<ShortestPathsResult> spDs = g.shortestPaths()
          .landmarks(Collections.singletonList(3L))
          .setAlgorithm("graphframes")
          .setUseLocalCheckpoints(true)
          .run();
      List<ShortestPathsResult> spGf = spDs.collectAsList();
      spDs.unpersist(true);
      assertEq(4, spGf.size(), "JavaShortestPaths graphframes count");
      Map<Long, Integer> d1 = spGf.stream()
          .filter(r -> r.id() == 1L)
          .findFirst()
          .orElseThrow(AssertionError::new)
          .distancesAsJava();
      assertEq(2, d1.get(3L), "distancesAsJava distance to 3");

      // JavaLabelPropagation graphframes algorithm with isolated vertex 4
      Dataset<LabelPropagationResult> lpDs = g.labelPropagation()
          .setAlgorithm("graphframes")
          .setUseLocalCheckpoints(true)
          .maxIter(3)
          .run();
      List<LabelPropagationResult> lpGf = lpDs.collectAsList();
      lpDs.unpersist(true);
      assertEq(4, lpGf.size(), "JavaLabelPropagation graphframes count");
      long isolatedLabel = lpGf.stream()
          .filter(r -> r.id() == 4L)
          .findFirst()
          .orElseThrow(AssertionError::new)
          .label();
      assertEq(4L, isolatedLabel, "isolated vertex keeps own ID label");

      // JavaConnectedComponents respects GraphFramesConf
      String algoKey = "spark.graphframes.connectedComponents.algorithm";
      scala.Option<String> prevAlgo = spark.conf().getOption(algoKey);
      spark.conf().set(algoKey, "randomized_contraction");
      try {
        Dataset<ConnectedComponentResult> ccRes =
            g.connectedComponents().setUseLocalCheckpoints(true).run();
        assertEq(4L, ccRes.count(), "JavaConnectedComponents with GraphFramesConf");
        ccRes.unpersist(true);
      } finally {
        if (prevAlgo.isDefined()) {
          spark.conf().set(algoKey, prevAlgo.get());
        } else {
          spark.conf().unset(algoKey);
        }
      }

      // JavaPregel preserves cached edges, restores job description, throws UncheckedIOException
      String confKey = "spark.sql.legacy.dataset.nameNonStructGroupingKeyAsValue";
      scala.Option<String> prevKeyConf = spark.conf().getOption(confKey);
      spark.conf().set(confKey, "true");
      spark.sparkContext().setJobDescription("caller-java-job");
      try {
        JavaGraphDataset<Integer, EdgeBean> pregelGraph =
            g.mapVertices((id, p) -> id == 1L ? 1 : 0, Encoders.INT());
        pregelGraph.edges().cache();
        pregelGraph.edges().count();
        JavaGraphDataset<Integer, EdgeBean> pregelResult = pregelGraph.pregel()
            .setMaxIter(2)
            .setCheckpointInterval(0)
            .setJobDescriptionPrefix("JavaCustomPregel")
            .setActiveDirection(EdgeDirection.Either())
            .withMessages(0, Encoders.INT())
            .vprog((id, attr, msg) -> Math.max(attr, msg))
            .sendMsgToDstOpt(t -> t.srcAttr() > t.dstAttr()
                ? Optional.of(t.srcAttr())
                : Optional.empty())
            .mergeMsg(Math::max)
            .run();
        assertEq("caller-java-job",
            spark.sparkContext().getLocalProperty("spark.job.description"),
            "job description restored");
        assertEq(false, StorageLevel.NONE().equals(pregelGraph.edges().storageLevel()),
            "JavaPregel preserves cached edges");
        pregelResult.vertices().unpersist(true);
        assertEq(false, StorageLevel.NONE().equals(pregelGraph.edges().storageLevel()),
            "unpersisting vertices preserves input cached edges");
        pregelGraph.edges().unpersist(true);
      } finally {
        spark.sparkContext().setJobDescription(null);
        if (prevKeyConf.isDefined()) {
          spark.conf().set(confKey, prevKeyConf.get());
        } else {
          spark.conf().unset(confKey);
        }
      }

      // Missing checkpoint dir throws UncheckedIOException in JavaPregel
      scala.Option<String> prevDir = spark.sparkContext().getCheckpointDir();
      assertEq(true, spark.conf().getOption("spark.checkpoint.dir").isEmpty(),
          "spark.checkpoint.dir should be unset in test env");
      spark.sparkContext().setCheckpointDir(null);
      boolean caughtUnchecked = false;
      try {
        g.mapVertices((id, p) -> 0, Encoders.INT())
            .pregel()
            .setCheckpointInterval(1)
            .setUseLocalCheckpoints(false)
            .withMessages(0, Encoders.INT())
            .vprog((id, attr, msg) -> attr)
            .sendMsgToDst(t -> t.srcAttr())
            .mergeMsg(Integer::sum)
            .run();
      } catch (UncheckedIOException expected) {
        caughtUnchecked = true;
      } finally {
        if (prevDir.isDefined()) {
          spark.sparkContext().setCheckpointDir(prevDir.get());
        }
      }
      assertEq(true, caughtUnchecked, "JavaPregel throws UncheckedIOException without ckptDir");

      // SVD++ Java list accessors & empty graph / self-loop checks
      Tuple2<Dataset<SVDPlusPlusResult>, Double> svd =
          g.svdPlusPlus().rank(4).maxIter(2).run(EdgeBean::getWeight);
      SVDPlusPlusResult firstSvd = svd._1().first();
      assertEq(4, firstSvd.column1AsJava().size(), "column1AsJava size");
      assertEq(4, firstSvd.column2AsJava().size(), "column2AsJava size");
      svd._1().unpersist(true);

      JavaGraphDataset<Integer, Integer> emptyJava = JavaGraphDataset.of(
          spark.emptyDataset(Encoders.tuple(Encoders.LONG(), Encoders.INT())),
          spark.emptyDataset(Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.INT())),
          Encoders.INT(),
          Encoders.INT());
      assertEq(0L, emptyJava.degrees().count(), "empty Java graph degrees");

      // JavaTriangleCount setLgNomEntries & eager setAlgorithm validation
      g.triangleCount().setLgNomEntries(8);
      boolean caughtTcAlgo = false;
      try {
        g.triangleCount().setAlgorithm("invalid");
      } catch (IllegalArgumentException expected) {
        caughtTcAlgo = true;
      }
      assertEq(true, caughtTcAlgo, "JavaTriangleCount rejects invalid algorithm eagerly");

      boolean caughtCcAlgo = false;
      try {
        g.connectedComponents().setAlgorithm("invalid");
      } catch (IllegalArgumentException expected) {
        caughtCcAlgo = true;
      }
      assertEq(true, caughtCcAlgo, "JavaConnectedComponents rejects invalid algorithm eagerly");

      // Java typeOutDegree / typeInDegree / typeDegree with edgeTypes filter
      List<Tuple3<Long, String, Integer>> outFiltered = g.typeOutDegree(
          EdgeBean::getKind, Collections.singletonList("friend"), Encoders.STRING())
          .collectAsList();
      assertEq(2, outFiltered.size(), "typeOutDegree filtered count");
      for (Tuple3<Long, String, Integer> row : outFiltered) {
        assertEq("friend", row._2(), "typeOutDegree filtered category");
      }

      List<Tuple3<Long, String, Integer>> inFiltered = g.typeInDegree(
          EdgeBean::getKind, Collections.singletonList("follow"), Encoders.STRING())
          .collectAsList();
      assertEq(1, inFiltered.size(), "typeInDegree filtered count");
      assertEq(3L, inFiltered.get(0)._1(), "typeInDegree filtered vertex");

      List<Tuple3<Long, String, Integer>> degFiltered = g.typeDegree(
          EdgeBean::getKind, Arrays.asList("friend", "nonexistent"), Encoders.STRING())
          .collectAsList();
      assertEq(3, degFiltered.size(), "typeDegree filtered count");

      List<Tuple3<Long, String, Integer>> degEmpty = g.typeDegree(
          EdgeBean::getKind, Collections.emptyList(), Encoders.STRING())
          .collectAsList();
      assertEq(0, degEmpty.size(), "typeDegree empty edgeTypes filter");
    } finally {
      g.unpersist(true);
    }
  }
}
