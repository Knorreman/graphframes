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

import org.apache.commons.io.FileUtils
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.hll_sketch_estimate
import org.graphframes.GraphDataset
import org.graphframes.GraphFrameTestSparkContext
import org.graphframes.GraphFramesSparkVersionException
import org.graphframes.SparkFunSuite
import org.graphframes.embeddings.Hash2Vec
import org.graphframes.rw.RandomWalkWithRestart

import java.nio.file.Files

case class CityInfo(name: String, pop: Int)
case class RoadInfo(highway: String, lanes: Int)

class GraphDatasetAlgorithmsSuite extends SparkFunSuite with GraphFrameTestSparkContext {

  import sqlImplicits.*

  test("pageRank preserves custom case class VD and ED") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, CityInfo("A", 100)),
        GraphVertex(2L, CityInfo("B", 200)),
        GraphVertex(3L, CityInfo("C", 300))))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, RoadInfo("I-1", 2)),
        GraphEdge(2L, 3L, RoadInfo("I-2", 4)),
        GraphEdge(3L, 1L, RoadInfo("I-3", 2))))
    val g = GraphDataset(vv, ee)

    val pr = g.pageRank.resetProbability(0.15).maxIter(5).run()
    val verts = pr.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(verts.keySet === Set(1L, 2L, 3L))
    assert(verts(1L).attr === CityInfo("A", 100))
    assert(verts(1L).pagerank > 0.0)

    val edges = pr.edges.collect().map(e => (e.srcId, e.dstId) -> e.attr).toMap
    assert(edges((1L, 2L)).attr === RoadInfo("I-1", 2))
    assert(edges((1L, 2L)).weight === 1.0)
  }

  test("shortestPaths") {
    val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i")))
    val ee =
      spark.createDataset(Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1), GraphEdge(3L, 4L, 1)))
    val g = GraphDataset(vv, ee)

    val sp = g.shortestPaths
      .landmarks(Seq(4L))
      .run()
      .collect()
      .map(r => r.id -> r.distances)
      .toMap

    assert(sp(1L) === Map(4L -> 3))
    assert(sp(2L) === Map(4L -> 2))
    assert(sp(3L) === Map(4L -> 1))
    assert(sp(4L) === Map(4L -> 0))
  }

  test("triangleCount") {
    val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(
      Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1), GraphEdge(3L, 1L, 1), GraphEdge(3L, 4L, 1)))
    val g = GraphDataset(vv, ee)

    val tc = g.triangleCount.run().collect().map(r => r.id -> r.count).toMap
    assert(tc === Map(1L -> 1L, 2L -> 1L, 3L -> 1L, 4L -> 0L))
  }

  test("labelPropagation") {
    // Two disjoint cliques of size 3 converge cleanly under synchronous LPA
    val vv = spark.createDataset((1L to 6L).map(i => GraphVertex(i, s"v$i")))
    val clique1 = for { i <- 1L to 3L; j <- 1L to 3L if i != j } yield GraphEdge(i, j, 1)
    val clique2 = for { i <- 4L to 6L; j <- 4L to 6L if i != j } yield GraphEdge(i, j, 1)
    val ee = spark.createDataset(clique1 ++ clique2)
    val g = GraphDataset(vv, ee)

    val lp = g.labelPropagation.maxIter(5).run().collect().map(r => r.id -> r.label).toMap
    assert(lp.keySet === (1L to 6L).toSet)
    assert(Set(lp(1L), lp(2L), lp(3L)).size === 1)
    assert(Set(lp(4L), lp(5L), lp(6L)).size === 1)
    assert(lp(1L) !== lp(4L))
  }

  test("stronglyConnectedComponents") {
    val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(
      Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 1L, 1), GraphEdge(2L, 3L, 1), GraphEdge(3L, 4L, 1)))
    val g = GraphDataset(vv, ee)

    val scc = g.stronglyConnectedComponents
      .maxIter(5)
      .run()
      .collect()
      .map(r => r.id -> r.connectedComponent)
      .toMap

    assert(scc(1L) === scc(2L))
    assert(scc(3L) !== scc(1L))
    assert(scc(4L) !== scc(3L))
  }

  test("kCore") {
    // Triangle (1, 2, 3) has 2-core; dangling vertex 4 has 1-core; isolated 5 has 0-core
    val vv = spark.createDataset((1L to 5L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(
      Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1), GraphEdge(1L, 3L, 1), GraphEdge(3L, 4L, 1)))
    val g = GraphDataset(vv, ee)

    val kc = g.kCore
      .setUseLocalCheckpoints(true)
      .run()
      .collect()
      .map(r => r.id -> r.kcore)
      .toMap

    assert(kc === Map(1L -> 2, 2L -> 2, 3L -> 2, 4L -> 1, 5L -> 0))
  }

  test("maximalIndependentSet preserves custom case class VD") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, CityInfo("A", 100)),
        GraphVertex(2L, CityInfo("B", 200)),
        GraphVertex(3L, CityInfo("C", 300))))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1)))
    val g = GraphDataset(vv, ee)

    val mis = g.maximalIndependentSet
      .setUseLocalCheckpoints(true)
      .run(seed = 42L)
      .collect()
    val misIds = mis.map(_.id).toSet
    // For path 1-2-3, any MIS is either {2} or {1, 3}
    assert(misIds === Set(2L) || misIds === Set(1L, 3L))
    mis.foreach { v =>
      assert(v.attr.pop === v.id.toInt * 100)
    }
  }

  test("detectingCycles") {
    val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(
      Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1), GraphEdge(3L, 1L, 1), GraphEdge(3L, 4L, 1)))
    val g = GraphDataset(vv, ee)

    val cycles = g.detectingCycles
      .setUseLocalCheckpoints(true)
      .run()
      .collect()
      .toSet

    assert(cycles === Set(Seq(1L, 2L, 3L, 1L)))
  }

  test("parallelPersonalizedPageRank preserves custom case class VD and ED") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, CityInfo("A", 100)),
        GraphVertex(2L, CityInfo("B", 200)),
        GraphVertex(3L, CityInfo("C", 300))))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, RoadInfo("I-1", 2)),
        GraphEdge(2L, 3L, RoadInfo("I-2", 4)),
        GraphEdge(3L, 1L, RoadInfo("I-3", 2))))
    val g = GraphDataset(vv, ee)

    val ppr = g.parallelPersonalizedPageRank
      .sourceIds(Seq(1L, 2L))
      .resetProbability(0.15)
      .maxIter(5)
      .run()

    val verts = ppr.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(verts.keySet === Set(1L, 2L, 3L))
    assert(verts(1L).attr === CityInfo("A", 100))
    assert(verts(1L).pageranks.size === 2)
    assert(verts(1L).pageranks(0) > verts(1L).pageranks(1))

    val edges = ppr.edges.collect().map(e => (e.srcId, e.dstId) -> e.attr).toMap
    assert(edges((1L, 2L)).attr === RoadInfo("I-1", 2))
    assert(edges((1L, 2L)).weight === 1.0)
  }

  test("hyperANF returns hop sketches and supports typed edge filtering") {
    val vv = spark.createDataset((1L to 5L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, "keep"),
        GraphEdge(1L, 3L, "keep"),
        GraphEdge(2L, 4L, "keep"),
        GraphEdge(3L, 4L, "keep"),
        GraphEdge(4L, 5L, "keep"),
        GraphEdge(5L, 1L, "keep")))
    val g = GraphDataset(vv, ee)

    val res = g.hyperANF.setNHops(2).setLgNomEntries(12).run()
    val collected = res.collect()
    assert(collected.map(_.id).toSet === (1L to 5L).toSet)
    collected.foreach(r => assert(r.numHops === 2))

    val estimates = res
      .select(
        col("id"),
        hll_sketch_estimate(col("hopSketches")(0)).alias("h0"),
        hll_sketch_estimate(col("hopSketches")(1)).alias("h1"),
        hll_sketch_estimate(col("hopSketches")(2)).alias("h2"))
      .collect()
      .map(r => r.getLong(0) -> Seq(r.getLong(1), r.getLong(2), r.getLong(3)))
      .toMap

    assert(
      estimates === Map(
        1L -> Seq(1L, 2L, 1L),
        2L -> Seq(1L, 1L, 1L),
        3L -> Seq(1L, 1L, 1L),
        4L -> Seq(1L, 1L, 1L),
        5L -> Seq(1L, 1L, 2L)))

    val filtered = g.hyperANF
      .setNHops(1)
      .filterEdges(_.srcId == 1L)
      .run()
      .collect()
    assert(filtered.map(_.id).toSet === Set(1L))
  }

  test("structureAwareLabelPropagation") {
    val vv = spark.createDataset((1L to 6L).map(i => GraphVertex(i, if (i <= 3L) "c1" else "c2")))
    val clique1 = for { i <- 1L to 3L; j <- 1L to 3L if i != j } yield GraphEdge(i, j, 1)
    val clique2 = for { i <- 4L to 6L; j <- 4L to 6L if i != j } yield GraphEdge(i, j, 1)
    val ee = spark.createDataset(clique1 ++ clique2)
    val g = GraphDataset(vv, ee)

    if (isLaterVersion("4.1")) {
      val salp = g.structureAwareLabelPropagation
        .maxIter(5)
        .setUseLocalCheckpoints(true)
        .run()
        .collect()
        .map(r => r.id -> r.label)
        .toMap
      assert(Set(salp(1L), salp(2L), salp(3L)).size === 1)
      assert(Set(salp(4L), salp(5L), salp(6L)).size === 1)
      assert(salp(1L) !== salp(4L))

      val attrProp = g.structureAwareLabelPropagation
        .maxIter(5)
        .setUseLocalCheckpoints(true)
        .runWithInitialLabel(_.attr)
        .collect()
        .map(v => v.id -> v.attr)
        .toMap
      assert(attrProp(1L) === "c1")
      assert(attrProp(4L) === "c2")
    } else {
      intercept[GraphFramesSparkVersionException] {
        g.structureAwareLabelPropagation.maxIter(5).run()
      }
    }
  }

  test("randomWalksBasedEmbedding") {
    val tmpDir = Files.createTempDirectory("gf-dataset-rw").toFile
    try {
      val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i")))
      val ee = spark.createDataset(
        Seq(
          GraphEdge(1L, 2L, 1),
          GraphEdge(2L, 3L, 1),
          GraphEdge(3L, 4L, 1),
          GraphEdge(4L, 1L, 1)))
      val g = GraphDataset(vv, ee)

      val rw = new RandomWalkWithRestart()
        .setNumWalksPerNode(2)
        .setBatchSize(4)
        .setNumBatches(2)
        .setRestartProbability(0.1)
        .setTemporaryPrefix(tmpDir.getAbsolutePath)
        .setGlobalSeed(42L)
      val h2v = new Hash2Vec()
        .setEmbeddingsDim(8)
        .setContextSize(2)
        .setNumPartitions(2)
        .setDoNormalization(doNorm = true, safeNorm = false)

      val emb = g.randomWalksBasedEmbedding
        .setRandomWalks(rw)
        .setSequenceModel(Right(h2v))
        .setAggregateNeighbors(false)
        .setCleanUpAfterRun(true)
        .run()
        .collect()

      assert(emb.map(_.id).toSet === (1L to 4L).toSet)
      emb.foreach(r => assert(r.embedding.size === 8))
    } finally {
      val _ = FileUtils.deleteQuietly(tmpDir)
    }
  }
}
