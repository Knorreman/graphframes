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
import org.apache.spark.storage.StorageLevel
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

  test("svdPlusPlus with typed rating extractor and numeric overload") {
    // Directed bipartite graph: users (1, 2) -> items (10, 11, 12)
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, "u1"),
        GraphVertex(2L, "u2"),
        GraphVertex(10L, "i1"),
        GraphVertex(11L, "i2"),
        GraphVertex(12L, "i3")))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 10L, RoadInfo("rate", 5)),
        GraphEdge(1L, 11L, RoadInfo("rate", 3)),
        GraphEdge(2L, 11L, RoadInfo("rate", 4)),
        GraphEdge(2L, 12L, RoadInfo("rate", 2))))
    val g = GraphDataset(vv, ee)

    val svd = g.svdPlusPlus.rank(4).maxIter(2)
    val (res, loss) = svd.run(_.lanes.toDouble)
    val collected = res.collect()
    assert(collected.map(_.id).toSet === Set(1L, 2L, 10L, 11L, 12L))
    collected.foreach { r =>
      assert(r.column1.size === 4)
      assert(r.column2.size === 4)
    }
    assert(loss >= 0.0)
    assert(svd.loss === loss)

    // Numeric ED convenience overload
    val numericGraph = g.mapEdges(_.attr.lanes.toDouble)
    val (res2, loss2) = numericGraph.svdPlusPlus.rank(4).maxIter(2).run()
    assert(res2.count() === 5L)
    assert(loss2 >= 0.0)
  }

  test("powerIterationClustering with and without typed weight extractor") {
    // Two strongly connected clusters (1-2-3) and (4-5-6) with a weak bridge (3-4)
    val vv = spark.createDataset((1L to 6L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, RoadInfo("strong", 10)),
        GraphEdge(2L, 3L, RoadInfo("strong", 10)),
        GraphEdge(1L, 3L, RoadInfo("strong", 10)),
        GraphEdge(4L, 5L, RoadInfo("strong", 10)),
        GraphEdge(5L, 6L, RoadInfo("strong", 10)),
        GraphEdge(4L, 6L, RoadInfo("strong", 10)),
        GraphEdge(3L, 4L, RoadInfo("weak", 1))))
    val g = GraphDataset(vv, ee)

    val weighted = g
      .powerIterationClustering(k = 2, maxIter = 20, weightExtractor = Some(_.lanes.toDouble))
      .collect()
      .map(c => c.id -> c.cluster)
      .toMap

    assert(weighted.keySet === (1L to 6L).toSet)
    assert(Set(weighted(1L), weighted(2L), weighted(3L)).size === 1)
    assert(Set(weighted(4L), weighted(5L), weighted(6L)).size === 1)
    assert(weighted(1L) !== weighted(4L))

    val unweighted = g.powerIterationClustering(k = 2, maxIter = 10).collect()
    assert(unweighted.map(_.id).toSet === (1L to 6L).toSet)
  }

  test("typeOutDegree, typeInDegree, and typeDegree with typed edge classifier") {
    val vv = spark.createDataset((1L to 3L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, RoadInfo("friend", 1)),
        GraphEdge(1L, 3L, RoadInfo("follow", 1)),
        GraphEdge(2L, 1L, RoadInfo("friend", 1)),
        GraphEdge(2L, 3L, RoadInfo("friend", 1))))
    val g = GraphDataset(vv, ee)

    val outDeg = g.typeOutDegree(_.highway).collect().map(r => r.id -> r.degrees).toMap
    assert(outDeg === Map(1L -> Map("friend" -> 1, "follow" -> 1), 2L -> Map("friend" -> 2)))

    val inDegWithTypes = g
      .typeInDegree(_.highway, edgeTypes = Some(Seq("friend", "follow", "block")))
      .collect()
      .map(r => r.id -> r.degrees)
      .toMap
    assert(
      inDegWithTypes === Map(
        1L -> Map("friend" -> 1, "follow" -> 0, "block" -> 0),
        2L -> Map("friend" -> 1, "follow" -> 0, "block" -> 0),
        3L -> Map("friend" -> 1, "follow" -> 1, "block" -> 0)))

    val totalDeg = g.typeDegree(_.highway).collect().map(r => r.id -> r.degrees).toMap
    assert(
      totalDeg === Map(
        1L -> Map("friend" -> 2, "follow" -> 1),
        2L -> Map("friend" -> 3),
        3L -> Map("friend" -> 1, "follow" -> 1)))
  }

  test("labelPropagation graphframes algorithm handles isolated vertices") {
    val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 1L, 1)))
    val g = GraphDataset(vv, ee)

    val lp = g.labelPropagation
      .setAlgorithm("graphframes")
      .setUseLocalCheckpoints(true)
      .maxIter(3)
      .run()
    try {
      val map = lp.collect().map(r => r.id -> r.label).toMap
      assert(map.keySet === Set(1L, 2L, 3L, 4L))
      assert(map(3L) === 3L)
      assert(map(4L) === 4L)
    } finally {
      val _ = lp.unpersist(blocking = true)
    }
  }

  test("type*Degree ignores null categories, works without Encoder implicit, and handles empty") {
    val vv = spark.createDataset((1L to 3L).map(i => GraphVertex(i, i)))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, null.asInstanceOf[String]),
        GraphEdge(1L, 3L, "friend"),
        GraphEdge(2L, 3L, "other")))
    val g = GraphDataset(vv, ee)

    val outNone =
      g.typeOutDegree(_.asInstanceOf[String]).collect().map(r => r.id -> r.degrees).toMap
    assert(outNone === Map(1L -> Map("friend" -> 1), 2L -> Map("other" -> 1)))

    val outFiltered = g
      .typeOutDegree(_.asInstanceOf[String], edgeTypes = Some(Seq("friend", "follow")))
      .collect()
      .map(r => r.id -> r.degrees)
      .toMap
    assert(
      outFiltered === Map(
        1L -> Map("friend" -> 1, "follow" -> 0),
        2L -> Map("friend" -> 0, "follow" -> 0)))

    val emptyG =
      GraphDataset(spark.emptyDataset[GraphVertex[Int]], spark.emptyDataset[GraphEdge[String]])
    assert(emptyG.typeOutDegree(_.toString).isEmpty)
  }

  test("wrapped algorithms release raw caches and PageRank supports tol & sourceId") {
    // Verify DatasetCacheUtils directly releases raw DataFrame cache while persisting typed output
    val rawDF = spark
      .createDataFrame(Seq((1L, 10L), (2L, 20L)))
      .toDF("id", "label")
      .persist(StorageLevel.MEMORY_AND_DISK)
    assert(rawDF.count() === 2L)
    assert(rawDF.storageLevel !== StorageLevel.NONE)
    val typedDS = rawDF.select(col("id"), col("label")).as[LabelPropagationResult]
    val cachedOut =
      DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typedDS, StorageLevel.MEMORY_AND_DISK)
    assert(rawDF.storageLevel === StorageLevel.NONE)
    assert(cachedOut.storageLevel !== StorageLevel.NONE)
    assert(cachedOut.collect().map(r => r.id -> r.label).toMap === Map(1L -> 10L, 2L -> 20L))
    cachedOut.unpersist(blocking = true)
    assert(cachedOut.storageLevel === StorageLevel.NONE)

    val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i"))).cache()
    val ee = spark
      .createDataset(Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1), GraphEdge(3L, 1L, 1)))
      .cache()
    assert(vv.count() === 4L)
    assert(ee.count() === 3L)
    val g = GraphDataset(vv, ee)

    try {
      // Pure DataFrame algorithms should leave zero non-checkpointed persistent RDDs after blocking unpersist
      var beforeIds = sc.getPersistentRDDs.keySet
      def assertNoLeaks(step: String): Unit = {
        val leaked = sc.getPersistentRDDs.filter { case (id, rdd) =>
          !beforeIds.contains(id) && !rdd.isCheckpointed
        }
        assert(
          leaked.isEmpty,
          s"Unexpected persistent RDDs after $step: ${leaked.values.map(_.name).mkString(", ")}")
        ()
      }

      val kc = g.kCore.setUseLocalCheckpoints(false).run()
      assert(kc.storageLevel !== StorageLevel.NONE)
      kc.unpersist(blocking = true)
      assertNoLeaks("kCore")

      val tc = g.triangleCount.run()
      assert(tc.storageLevel !== StorageLevel.NONE)
      tc.unpersist(blocking = true)
      assertNoLeaks("triangleCount")

      val spGf = g.shortestPaths
        .landmarks(Seq(1L))
        .setAlgorithm("graphframes")
        .setUseLocalCheckpoints(false)
        .run()
      assert(spGf.storageLevel !== StorageLevel.NONE)
      spGf.unpersist(blocking = true)
      assertNoLeaks("shortestPaths")

      // MaximalIndependentSet upstream caches internal intermediate DataFrames before returning raw DF;
      // verify our typed wrapper dataset itself is persisted and cleanly unpersists, then re-baseline.
      val mis = g.maximalIndependentSet.setUseLocalCheckpoints(false).run(42L)
      assert(mis.storageLevel !== StorageLevel.NONE)
      mis.unpersist(blocking = true)
      assert(mis.storageLevel === StorageLevel.NONE)
      beforeIds = sc.getPersistentRDDs.keySet

      val dc = g.detectingCycles.setUseLocalCheckpoints(false).run()
      assert(dc.storageLevel !== StorageLevel.NONE)
      dc.unpersist(blocking = true)
      assertNoLeaks("detectingCycles")

      // PageRank with tol and sourceId
      val ppr = g.pageRank.sourceId(1L).tol(0.01).run()
      assert(ppr.vertices.storageLevel !== StorageLevel.NONE)
      assert(ppr.edges.storageLevel !== StorageLevel.NONE)
      val scores = ppr.vertices.collect().map(v => v.id -> v.attr.pagerank).toMap
      assert(scores(1L) > scores(4L))
      ppr.unpersist(blocking = true)
      assert(ppr.vertices.storageLevel === StorageLevel.NONE)
      assert(ppr.edges.storageLevel === StorageLevel.NONE)
    } finally {
      vv.unpersist(blocking = true)
      ee.unpersist(blocking = true)
      ()
    }
  }

  test("RandomWalkEmbeddings validation and HyperANFResult numHops guard") {
    val vv = spark.createDataset(Seq(GraphVertex(1L, "a")))
    val ee = spark.emptyDataset[GraphEdge[Int]]
    val g = GraphDataset(vv, ee)

    intercept[IllegalArgumentException] {
      val _ = g.randomWalksBasedEmbedding.run()
    }
    assert(HyperANFResult(1L, Seq.empty).numHops === 0)
  }
}
