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

package org.graphframes.lib

import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphDataset
import org.graphframes.GraphFrameTestSparkContext
import org.graphframes.SparkFunSuite

case class Station(name: String, zone: Int)
case class Track(kind: String, distance: Int)
case class RouteAcc(visited: Seq[VertexId], totalDist: Int)

class GraphDatasetTraversalsSuite extends SparkFunSuite with GraphFrameTestSparkContext {

  import sqlImplicits.*

  test(
    "bfs finds shortest typed paths with custom case classes, edge filter, 0-hop, and no-path") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, Station("A", 1)),
        GraphVertex(2L, Station("B", 1)),
        GraphVertex(3L, Station("C", 2)),
        GraphVertex(4L, Station("D", 2)),
        GraphVertex(5L, Station("Isolated", 3))))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, Track("rail", 5)),
        GraphEdge(2L, 4L, Track("rail", 6)),
        GraphEdge(1L, 3L, Track("bus", 3)),
        GraphEdge(3L, 4L, Track("bus", 4))))
    val g = GraphDataset(vv, ee)

    // Two shortest paths of length 2 from A to D
    val paths = g.bfs
      .fromPred(_.attr.name == "A")
      .toPred(_.attr.name == "D")
      .run()
      .collect()
    assert(paths.length === 2)
    paths.foreach { p =>
      assert(p.length === 2)
      assert(p.from === GraphVertex(1L, Station("A", 1)))
      assert(p.to === GraphVertex(4L, Station("D", 2)))
    }
    val vertexSeqs = paths.map(_.vertices.map(_.id)).toSet
    assert(vertexSeqs === Set(Seq(1L, 2L, 4L), Seq(1L, 3L, 4L)))

    // With typed edgeFilter restricting to "rail"
    val railOnly = g.bfs
      .fromPred(_.attr.name == "A")
      .toPred(_.attr.name == "D")
      .edgeFilter(_.attr.kind == "rail")
      .run()
      .collect()
    assert(railOnly.length === 1)
    assert(railOnly.head.vertices.map(_.id) === Seq(1L, 2L, 4L))
    assert(railOnly.head.edges.map(_.attr) === Seq(Track("rail", 5), Track("rail", 6)))

    // 0-hop path (from == to)
    val zeroHop = g.bfs
      .fromPred(_.id == 1L)
      .toPred(_.attr.zone == 1)
      .run()
      .collect()
    assert(zeroHop.length === 1)
    assert(zeroHop.head.length === 0)
    assert(zeroHop.head.from === GraphVertex(1L, Station("A", 1)))
    assert(zeroHop.head.to === GraphVertex(1L, Station("A", 1)))
    assert(zeroHop.head.edges.isEmpty)

    // No path found
    val noPath = g.bfs
      .fromPred(_.id == 1L)
      .toPred(_.id == 5L)
      .maxPathLength(3)
      .run()
      .collect()
    assert(noPath.isEmpty)
  }

  test("allPaths directed, undirected, and with typed edgeFilter") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, Station("A", 1)),
        GraphVertex(2L, Station("B", 1)),
        GraphVertex(3L, Station("C", 2)),
        GraphVertex(4L, Station("D", 2))))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, Track("rail", 5)),
        GraphEdge(2L, 3L, Track("rail", 4)),
        GraphEdge(3L, 4L, Track("rail", 3)),
        GraphEdge(1L, 3L, Track("bus", 7)),
        GraphEdge(2L, 1L, Track("rail", 5))
      )
    ) // back-edge to ensure simple paths only
    val g = GraphDataset(vv, ee)

    val directed = g.allPaths
      .fromPred(_.attr.name == "A")
      .toPred(_.attr.name == "D")
      .maxPathLength(4)
      .setUseLocalCheckpoints(true)
      .run()
      .collect()
      .toSet

    assert(directed === Set(SimplePath(Seq(1L, 2L, 3L, 4L), 3), SimplePath(Seq(1L, 3L, 4L), 2)))

    val railOnly = g.allPaths
      .fromPred(_.id == 1L)
      .toPred(_.id == 4L)
      .edgeFilter(_.attr.kind == "rail")
      .maxPathLength(4)
      .setUseLocalCheckpoints(true)
      .run()
      .collect()
      .toSet

    assert(railOnly === Set(SimplePath(Seq(1L, 2L, 3L, 4L), 3)))

    // Undirected path from 4 back to 1
    val undirected = g.allPaths
      .fromPred(_.id == 4L)
      .toPred(_.id == 1L)
      .setIsDirected(false)
      .maxPathLength(2)
      .setUseLocalCheckpoints(true)
      .run()
      .collect()
      .toSet

    assert(undirected === Set(SimplePath(Seq(4L, 3L, 1L), 2)))
  }

  test("aggregateNeighbors with custom case class accumulator, target and stopping conditions") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, Station("A", 1)),
        GraphVertex(2L, Station("B", 1)),
        GraphVertex(3L, Station("C", 2)),
        GraphVertex(4L, Station("D", 2))))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, Track("rail", 5)),
        GraphEdge(2L, 3L, Track("rail", 4)),
        GraphEdge(3L, 4L, Track("rail", 3)),
        GraphEdge(1L, 3L, Track("bus", 7)),
        GraphEdge(2L, 1L, Track("rail", 5))
      )
    ) // cycle back to 1
    val g = GraphDataset(vv, ee)

    val res = g.aggregateNeighbors
      .setStartingVertices(_.id == 1L)
      .setMaxHops(4)
      .setCheckpointInterval(2)
      .setUseLocalCheckpoints(true)
      .withAccumulator[RouteAcc](
        init = v => RouteAcc(Seq(v.id), 0),
        update = (acc, t) => RouteAcc(acc.visited :+ t.dstId, acc.totalDist + t.attr.distance))
      .setStoppingCondition((acc, t) => acc.visited.contains(t.dstId))
      .setTargetCondition((_, t) => t.dstAttr.name == "D")
      .run()
      .collect()
      .toSet

    assert(
      res === Set(
        NeighborAggregateResult(4L, 3, RouteAcc(Seq(1L, 2L, 3L, 4L), 12)),
        NeighborAggregateResult(4L, 2, RouteAcc(Seq(1L, 3L, 4L), 10))))
  }

  test("aggregateNeighbors collects stopped paths when no targetCondition is set") {
    val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, i.toInt)))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, 10),
        GraphEdge(2L, 3L, 20),
        GraphEdge(3L, 4L, 30),
        GraphEdge(1L, 1L, 5)))
    val g = GraphDataset(vv, ee)

    val res = g.aggregateNeighbors
      .setStartingVertices(_.id == 1L)
      .setMaxHops(5)
      .setRemoveLoops(true)
      .setCheckpointInterval(0)
      .withAccumulator[Int](init = _ => 0, update = (sum, t) => sum + t.attr)
      .setStoppingCondition((sum, t) => sum + t.attr >= 25)
      .run()
      .collect()

    assert(res.toSet === Set(NeighborAggregateResult(3L, 2, 30)))
  }

  test("bfs 0-hop works with legacy createEmptyCollectionUsingStringType conf") {
    val confKey = "spark.sql.legacy.createEmptyCollectionUsingStringType"
    val prev = spark.conf.getOption(confKey)
    spark.conf.set(confKey, "true")
    try {
      val vv = spark.createDataset(Seq(GraphVertex(1L, Station("A", 1))))
      val ee = spark.emptyDataset[GraphEdge[Track]]
      val g = GraphDataset(vv, ee)

      val zeroHop = g.bfs.fromPred(_.id == 1L).toPred(_.id == 1L).run().collect()
      assert(zeroHop.length === 1)
      assert(zeroHop.head.length === 0)
      assert(zeroHop.head.edges.isEmpty)
    } finally {
      prev match {
        case Some(v) => spark.conf.set(confKey, v)
        case None => spark.conf.unset(confKey)
      }
    }
  }

  test("empty graph traversals, parallel edges dedup, and aggregateNeighbors cache cleanup") {
    val emptyV = spark.emptyDataset[GraphVertex[Station]]
    val emptyE = spark.emptyDataset[GraphEdge[Track]]
    val emptyG = GraphDataset(emptyV, emptyE)

    assert(emptyG.bfs.fromPred(_.id == 1L).toPred(_.id == 2L).run().isEmpty)
    assert(
      emptyG.allPaths
        .fromPred(_.id == 1L)
        .toPred(_.id == 2L)
        .setUseLocalCheckpoints(true)
        .run()
        .isEmpty)

    // Parallel edges should produce a single SimplePath per topological route
    val vv =
      spark.createDataset(Seq(GraphVertex(1L, Station("A", 1)), GraphVertex(2L, Station("B", 1))))
    val parallelE = spark.createDataset(
      Seq(GraphEdge(1L, 2L, Track("rail", 1)), GraphEdge(1L, 2L, Track("bus", 2))))
    val pg = GraphDataset(vv, parallelE)
    val paths = pg.allPaths
      .fromPred(_.id == 1L)
      .toPred(_.id == 2L)
      .setUseLocalCheckpoints(true)
      .run()
      .collect()
    assert(paths.length === 1)

    // Verify aggregateNeighbors returns persisted result and unpersist leaves zero leaked RDDs
    val cachedV = vv.cache()
    val cachedE = parallelE.cache()
    val _ = (cachedV.count(), cachedE.count())
    val beforeIds = sc.getPersistentRDDs.keySet.toSet
    val cg = GraphDataset(cachedV, cachedE)
    try {
      val aggRes = cg.aggregateNeighbors
        .setStartingVertices(_.id == 1L)
        .setMaxHops(2)
        .setCheckpointInterval(0)
        .withAccumulator[Int](_ => 0, (s, t) => s + t.attr.distance)
        .setTargetCondition((_, t) => t.dstId == 2L)
        .run()
      assert(aggRes.storageLevel !== StorageLevel.NONE)
      assert(aggRes.count() === 2L)
      aggRes.unpersist(blocking = true)
      assert(sc.getPersistentRDDs.keySet.filterNot(beforeIds.contains).isEmpty)
    } finally {
      val _ = (cachedV.unpersist(blocking = true), cachedE.unpersist(blocking = true))
    }
  }
}
