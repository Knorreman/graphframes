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

import org.apache.spark.graphframes.graphx.EdgeDirection;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;
import org.graphframes.lib.ClusterAssignment;
import org.graphframes.lib.ConnectedComponentResult;
import org.graphframes.lib.KCoreResult;
import org.graphframes.lib.LabelPropagationResult;
import org.graphframes.lib.SVDPlusPlusResult;
import org.graphframes.lib.ShortestPathsResult;
import org.graphframes.lib.SimplePath;
import org.graphframes.lib.TriangleCountResult;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple5;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
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
    JavaGraphDataset<String, EdgeBean> fromE = JavaGraphDataset.fromEdges(
        g.edges(), "default", Encoders.STRING(), Encoders.bean(EdgeBean.class));
    assertEq(3L, fromE.vertices().count(), "fromEdges vertex count");

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

  public static void testAggregateMessagesAndPregel(SparkSession spark) throws IOException {
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
}
