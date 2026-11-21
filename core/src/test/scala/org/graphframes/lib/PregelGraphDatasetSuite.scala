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

import org.apache.spark.graphframes.graphx.EdgeDirection
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphDataset
import org.graphframes.GraphFrameTestSparkContext
import org.graphframes.SparkFunSuite
import org.scalactic.Tolerance.*

import java.io.IOException

case class PathMsg(dist: Int, via: VertexId)
case class NodeState(label: String, dist: Int)
case class WeightedLink(kind: String, cost: Int)

class PregelGraphDatasetSuite extends SparkFunSuite with GraphFrameTestSparkContext {

  import sqlImplicits.*

  test("page rank") {
    // 4-node strongly connected graph: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0, and 0 -> 2
    val edges = spark.createDataset(
      Seq(
        GraphEdge(0L, 1L, 0.5),
        GraphEdge(0L, 2L, 0.5),
        GraphEdge(1L, 2L, 1.0),
        GraphEdge(2L, 3L, 1.0),
        GraphEdge(3L, 0L, 1.0)))
    val vertices = spark.createDataset((0L to 3L).map(id => GraphVertex(id, 0.0)))
    val graph = GraphDataset(vertices, edges)
    val numVertices = 4.0
    val alpha = 0.15

    val ranks = graph.pregel
      .setMaxIter(5)
      .setCheckpointInterval(2)
      .run[Double](0.0)(
        vprog = (_, attr, msg) =>
          if (attr == 0.0) 1.0 / numVertices
          else msg * (1.0 - alpha) + alpha / numVertices,
        sendMsg = triplet => Iterator((triplet.dstId, triplet.srcAttr * triplet.attr)),
        mergeMsg = (a, b) => a + b)

    val result = ranks.vertices
      .collect()
      .sortBy(_.id)
      .map(_.attr)

    assert(result.sum === 1.0 +- 1e-6)
    // Node 2 receives links from both 0 and 1, so it has higher rank than node 1
    assert(result(2) > result(1))
  }

  test("chain propagation with early termination") {
    val n = 5L
    val vv = spark.createDataset((1L to n).map(i => GraphVertex(i, if (i == 1L) 1 else 0)))
    val ee = spark.createDataset((1L until n).map(i => GraphEdge(i, i + 1L, 1)))
    val g = GraphDataset(vv, ee)

    val result = g.pregel
      .setMaxIter(100)
      .setCheckpointInterval(2)
      .setActiveDirection(EdgeDirection.Out)
      .run[Int](0)(
        vprog = (_, attr, msg) => math.max(attr, msg),
        sendMsg = t =>
          if (t.srcAttr > t.dstAttr) Iterator((t.dstId, t.srcAttr))
          else Iterator.empty,
        mergeMsg = (a, b) => math.max(a, b))

    val values = result.vertices.collect().sortBy(_.id).map(_.attr)
    assert(values === Array.fill(n.toInt)(1))
  }

  test("reverse chain propagation with EdgeDirection.In") {
    val n = 5L
    val vv = spark.createDataset((1L to n).map(i => GraphVertex(i, if (i == 1L) 1 else 0)))
    val ee = spark.createDataset((1L until n).map(i => GraphEdge(i + 1L, i, 1)))
    val g = GraphDataset(vv, ee)

    val result = g.pregel
      .setMaxIter(10)
      .setUseLocalCheckpoints(true)
      .setCheckpointInterval(1)
      .setActiveDirection(EdgeDirection.In)
      .run[Int](0)(
        vprog = (_, attr, msg) => math.max(attr, msg),
        sendMsg = t =>
          if (t.dstAttr > t.srcAttr) Iterator((t.srcId, t.dstAttr))
          else Iterator.empty,
        mergeMsg = (a, b) => math.max(a, b))

    val values = result.vertices.collect().sortBy(_.id).map(_.attr)
    assert(values === Array.fill(n.toInt)(1))
  }

  test("EdgeDirection.Both only activates edges where both endpoints received messages") {
    val vv = spark.createDataset(Seq(GraphVertex(1L, 1), GraphVertex(2L, 0), GraphVertex(3L, 0)))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1)))
    val g = GraphDataset(vv, ee)

    val result = g.pregel
      .setMaxIter(5)
      .setCheckpointInterval(0)
      .setActiveDirection(EdgeDirection.Both)
      .run[Int](0)(
        vprog = (_, attr, msg) => math.max(attr, msg),
        sendMsg = t =>
          if (t.srcAttr > t.dstAttr) Iterator((t.dstId, t.srcAttr))
          else Iterator.empty,
        mergeMsg = (a, b) => math.max(a, b))

    // Iter 0: all active -> 1 sends 1 to 2 (only vertex 2 receives a message)
    // Iter 1: Edge (2, 3) has vertex 2 active, vertex 3 inactive -> skipped under EdgeDirection.Both
    val values = result.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(values === Map(1L -> 1, 2L -> 1, 3L -> 0))
  }

  test("single-source shortest paths with custom case class message") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, 0),
        GraphVertex(2L, Int.MaxValue),
        GraphVertex(3L, Int.MaxValue),
        GraphVertex(4L, Int.MaxValue),
        GraphVertex(5L, Int.MaxValue)))
    val ee = spark.createDataset(
      Seq(GraphEdge(1L, 2L, 2), GraphEdge(1L, 3L, 5), GraphEdge(2L, 3L, 1), GraphEdge(3L, 4L, 3)))
    val g = GraphDataset(vv, ee)

    val result = PregelGraphDataset.run[Int, Int, PathMsg](
      graph = g,
      initialMsg = PathMsg(Int.MaxValue, -1L),
      maxIterations = 10,
      checkpointInterval = 2,
      activeDirection = EdgeDirection.Out,
      vprog = (_, dist, msg) => math.min(dist, msg.dist),
      sendMsg = t =>
        if (t.srcAttr != Int.MaxValue && t.srcAttr + t.attr < t.dstAttr) {
          Iterator((t.dstId, PathMsg(t.srcAttr + t.attr, t.srcId)))
        } else {
          Iterator.empty
        },
      mergeMsg = (a, b) => if (a.dist <= b.dist) a else b)

    val dists = result.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(dists === Map(1L -> 0, 2L -> 2, 3L -> 3, 4L -> 6, 5L -> Int.MaxValue))
  }

  test("fluent withMessages builder API") {
    val n = 5L
    val vv = spark.createDataset((1L to n).map(i => GraphVertex(i, if (i == 1L) 1 else 0)))
    val ee = spark.createDataset((1L until n).map(i => GraphEdge(i, i + 1L, 1)))
    val g = GraphDataset(vv, ee)

    val result = g.pregel
      .setMaxIter(10)
      .setCheckpointInterval(2)
      .setActiveDirection(EdgeDirection.Out)
      .withMessages[Int](0)
      .vprog((_, attr, msg) => math.max(attr, msg))
      .sendMsgToDstOpt(t => if (t.srcAttr > t.dstAttr) Some(t.srcAttr) else None)
      .mergeMsg(math.max)
      .run()

    val values = result.vertices.collect().sortBy(_.id).map(_.attr)
    assert(values === Array.fill(n.toInt)(1))
  }

  test("runTriplets using GraphTriplet directly") {
    val vv =
      spark.createDataset(Seq(GraphVertex(1L, 0), GraphVertex(2L, 100), GraphVertex(3L, 100)))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, 5), GraphEdge(2L, 3L, 7)))
    val g = GraphDataset(vv, ee)

    val result = g.pregel
      .setMaxIter(5)
      .setCheckpointInterval(0)
      .setActiveDirection(EdgeDirection.Out)
      .runTriplets[Int](100)(
        vprog = (_, dist, msg) => math.min(dist, msg),
        sendMsg = t =>
          if (t.srcAttr + t.attr < t.dstAttr) Iterator((t.dstId, t.srcAttr + t.attr))
          else Iterator.empty,
        mergeMsg = math.min)

    val dists = result.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(dists === Map(1L -> 0, 2L -> 5, 3L -> 12))
  }

  test("empty edges applies initialMsg and restores job description") {
    val vv = spark.createDataset(Seq(GraphVertex(1L, 10), GraphVertex(2L, 20)))
    val ee = spark.emptyDataset[GraphEdge[Int]]
    val g = GraphDataset(vv, ee)

    sc.setJobDescription("caller-job")
    val result = g.pregel
      .setMaxIter(5)
      .setCheckpointInterval(0)
      .setJobDescriptionPrefix("CustomTypedPregel")
      .withMessages[Int](5)
      .vprog((_, attr, msg) => attr + msg)
      .sendMsgToDst(_.srcAttr)
      .mergeMsg(_ + _)
      .run()

    assert(sc.getLocalProperty("spark.job.description") === "caller-job")
    assert(result.vertices.collect().map(v => v.id -> v.attr).toMap === Map(1L -> 15, 2L -> 25))
    result.vertices.unpersist()
    sc.setJobDescription(null)
  }

  test("custom case class VD, ED, and message types deserialize cleanly") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, NodeState("start", 0)),
        GraphVertex(2L, NodeState("mid", Int.MaxValue)),
        GraphVertex(3L, NodeState("end", Int.MaxValue))))
    val ee = spark.createDataset(
      Seq(GraphEdge(1L, 2L, WeightedLink("road", 4)), GraphEdge(2L, 3L, WeightedLink("rail", 6))))
    val g = GraphDataset(vv, ee)

    val result = g.pregel
      .setMaxIter(5)
      .setCheckpointInterval(2)
      .setActiveDirection(EdgeDirection.Out)
      .withMessages[PathMsg](PathMsg(Int.MaxValue, -1L))
      .vprog((_, state, msg) => if (msg.dist < state.dist) state.copy(dist = msg.dist) else state)
      .sendMsgToDstOpt(t =>
        if (t.srcAttr.dist != Int.MaxValue && t.srcAttr.dist + t.attr.cost < t.dstAttr.dist) {
          Some(PathMsg(t.srcAttr.dist + t.attr.cost, t.srcId))
        } else {
          None
        })
      .mergeMsg((a, b) => if (a.dist <= b.dist) a else b)
      .run()

    val states = result.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(
      states === Map(
        1L -> NodeState("start", 0),
        2L -> NodeState("mid", 4),
        3L -> NodeState("end", 10)))
    result.vertices.unpersist()
  }

  test("preserves pre-cached caller edges and unpersisting result vertices frees all caches") {
    val vv =
      spark.createDataset((1L to 4L).map(i => GraphVertex(i, if (i == 1L) 1 else 0))).cache()
    val ee = spark.createDataset((1L until 4L).map(i => GraphEdge(i, i + 1L, 1))).cache()
    val _ = (vv.count(), ee.count())
    val beforeIds = sc.getPersistentRDDs.keySet.toSet
    val g = GraphDataset(vv, ee)

    try {
      val result = g.pregel
        .setMaxIter(10)
        .setCheckpointInterval(0)
        .setActiveDirection(EdgeDirection.Out)
        .withMessages[Int](0)
        .vprog((_, attr, msg) => math.max(attr, msg))
        .sendMsgToDstOpt(t => if (t.srcAttr > t.dstAttr) Some(t.srcAttr) else None)
        .mergeMsg(math.max)
        .run()

      assert(g.edges.storageLevel !== StorageLevel.NONE)
      assert(result.vertices.storageLevel !== StorageLevel.NONE)
      assert(result.vertices.collect().forall(_.attr == 1))

      result.vertices.unpersist(blocking = true)
      assert(g.edges.storageLevel !== StorageLevel.NONE)
      assert(sc.getPersistentRDDs.keySet.filterNot(beforeIds.contains).isEmpty)
    } finally {
      vv.unpersist(blocking = true)
      ee.unpersist(blocking = true)
    }
  }

  test("works with legacy nameNonStructGroupingKeyAsValue conf") {
    val confKey = "spark.sql.legacy.dataset.nameNonStructGroupingKeyAsValue"
    val prev = spark.conf.getOption(confKey)
    spark.conf.set(confKey, "true")
    try {
      val vv = spark.createDataset(Seq(GraphVertex(1L, 10), GraphVertex(2L, 0)))
      val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, 1)))
      val g = GraphDataset(vv, ee)

      val result = g.pregel
        .setMaxIter(3)
        .setCheckpointInterval(0)
        .withMessages[Int](0)
        .vprog((_, attr, msg) => math.max(attr, msg))
        .sendMsgToDstOpt(t => if (t.srcAttr > t.dstAttr) Some(t.srcAttr) else None)
        .mergeMsg(math.max)
        .run()

      val map = result.vertices.collect().map(v => v.id -> v.attr).toMap
      assert(map === Map(1L -> 10, 2L -> 10))
      result.vertices.unpersist(blocking = true)
    } finally {
      prev match {
        case Some(v) => spark.conf.set(confKey, v)
        case None => spark.conf.unset(confKey)
      }
    }
  }

  test("EdgeDirection.Either, maxIter cap, self-loops, parallel edges, and empty graph") {
    val emptyV = spark.emptyDataset[GraphVertex[Int]]
    val emptyE = spark.emptyDataset[GraphEdge[Int]]
    val emptyRes = GraphDataset(emptyV, emptyE).pregel
      .setMaxIter(3)
      .setCheckpointInterval(0)
      .withMessages[Int](0)
      .vprog((_, attr, msg) => attr + msg)
      .sendMsgToDst(_.srcAttr)
      .mergeMsg(_ + _)
      .run()
    assert(emptyRes.vertices.isEmpty)
    emptyRes.vertices.unpersist(blocking = true)

    val vv = spark.createDataset(
      Seq(GraphVertex(1L, 1), GraphVertex(2L, 0), GraphVertex(3L, 0), GraphVertex(4L, 0)))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 1L, 1),
        GraphEdge(1L, 2L, 1),
        GraphEdge(1L, 2L, 2),
        GraphEdge(2L, 3L, 1),
        GraphEdge(3L, 4L, 1)))
    val g = GraphDataset(vv, ee)

    // With EdgeDirection.Either and maxIter(2), signal from 1 reaches 2 and 3, stopping before 4
    val capped = g.pregel
      .setMaxIter(2)
      .setCheckpointInterval(1)
      .setUseLocalCheckpoints(true)
      .setActiveDirection(EdgeDirection.Either)
      .withMessages[Int](0)
      .vprog((_, attr, msg) => math.max(attr, msg))
      .sendMsgToDstOpt(t => if (t.srcAttr > t.dstAttr) Some(t.srcAttr) else None)
      .mergeMsg(math.max)
      .run()

    val values = capped.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(values === Map(1L -> 1, 2L -> 1, 3L -> 1, 4L -> 0))
    capped.vertices.unpersist(blocking = true)
  }

  test("parameter validation and missing checkpointDir error") {
    val vv = spark.createDataset(Seq(GraphVertex(1L, 1)))
    val ee = spark.emptyDataset[GraphEdge[Int]]
    val g = GraphDataset(vv, ee)

    intercept[IllegalArgumentException] {
      val _ = g.pregel.setCheckpointInterval(-1)
    }
    intercept[IllegalArgumentException] {
      val _ = g.pregel.setActiveDirection(null)
    }

    val prevDir = sc.getCheckpointDir
    assert(spark.conf.getOption("spark.checkpoint.dir").isEmpty)
    sc.setCheckpointDir(null)
    try {
      intercept[IOException] {
        val _ = g.pregel
          .setCheckpointInterval(1)
          .setUseLocalCheckpoints(false)
          .withMessages[Int](0)
          .vprog((_, attr, _) => attr)
          .sendMsgToDst(_.srcAttr)
          .mergeMsg(_ + _)
          .run()
      }
    } finally {
      prevDir.foreach(sc.setCheckpointDir)
    }
  }
}
