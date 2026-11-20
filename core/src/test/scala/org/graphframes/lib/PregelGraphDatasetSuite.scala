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
import org.graphframes.GraphDataset
import org.graphframes.GraphFrameTestSparkContext
import org.graphframes.SparkFunSuite
import org.scalactic.Tolerance.*

case class PathMsg(dist: Int, via: VertexId)

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
}
