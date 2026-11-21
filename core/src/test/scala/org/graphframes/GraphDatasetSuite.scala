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

package org.graphframes

import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel
import org.graphframes.lib.GraphEdge
import org.graphframes.lib.GraphTriplet
import org.graphframes.lib.GraphVertex

import java.io.File
import java.nio.file.Files

case class PersonAttr(name: String, age: Int)
case class RelationAttr(kind: String, weight: Double)

class GraphDatasetSuite extends SparkFunSuite with GraphFrameTestSparkContext {
  val localVertices: Map[Long, String] = Map(1L -> "A", 2L -> "B", 3L -> "C")
  val localEdges: Map[(Long, Long), String] =
    Map((1L, 2L) -> "love", (2L, 1L) -> "hate", (2L, 3L) -> "follow")
  var edges: Dataset[GraphEdge[String]] = _
  var vertices: Dataset[GraphVertex[String]] = _
  var tempDir: File = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    import sqlImplicits.*
    tempDir = Files.createTempDirectory(this.getClass.getName).toFile
    vertices = spark.createDataset(localVertices.toSeq.map(tpl => GraphVertex(tpl._1, tpl._2)))
    edges = spark.createDataset(localEdges.toSeq.map { case ((src, dst), action) =>
      GraphEdge(src, dst, action)
    })
  }

  override def afterAll(): Unit = {
    FileUtils.deleteQuietly(tempDir)
    super.afterAll()
  }

  test("save/load") {
    import sqlImplicits.*
    val g0 = GraphDataset(vertices, edges)
    val vPath = new Path(tempDir.getPath, "vertices").toString
    val ePath = new Path(tempDir.getPath, "edges").toString
    g0.vertices.write.parquet(vPath)
    g0.edges.write.parquet(ePath)

    val v1 = spark.read.parquet(vPath).as[GraphVertex[String]]
    val e1 = spark.read.parquet(ePath).as[GraphEdge[String]]
    val g1 = GraphDataset(v1, e1)

    g1.vertices.collect().foreach { case GraphVertex(id: Long, name: String) =>
      assert(localVertices(id) === name)
    }
    g1.edges.collect().foreach { case GraphEdge(src: Long, dst: Long, action: String) =>
      assert(localEdges((src, dst)) === action)
    }
  }

  test("degree metrics") {
    val g: GraphDataset[String, String] = GraphDataset(vertices, edges)

    val outDegrees = g.outDegrees
      .collect()
      .map { case (id: Long, outDeg: Int) =>
        (id, outDeg)
      }
      .toMap
    assert(outDegrees === Map(1L -> 1, 2L -> 2))

    val inDegrees = g.inDegrees
      .collect()
      .map { case (id: Long, inDeg: Int) =>
        (id, inDeg)
      }
      .toMap
    assert(inDegrees === Map(1L -> 1, 2L -> 1, 3L -> 1))

    val degrees = g.degrees
      .collect()
      .map { case (id: Long, deg: Int) =>
        (id, deg)
      }
      .toMap
    assert(degrees === Map(1L -> 2, 2L -> 3, 3L -> 1))
  }

  test("toString and persist/unpersist") {
    val g = GraphDataset(vertices, edges)
    assert(g.toString.startsWith("GraphDataset(v:"))
    assert(g.toString.contains(", e:"))
    g.cache()
    assert(g.vertices.storageLevel !== StorageLevel.NONE)
    assert(g.edges.storageLevel !== StorageLevel.NONE)
    g.unpersist(blocking = true)
    assert(g.vertices.storageLevel === StorageLevel.NONE)
    assert(g.edges.storageLevel === StorageLevel.NONE)
  }

  test("fromEdges") {
    val g = GraphDataset.fromEdges(edges, defaultVertexAttribute = "default")
    val vMap = g.vertices.collect().map(v => v.id -> v.attr).toMap
    assert(vMap === Map(1L -> "default", 2L -> "default", 3L -> "default"))
    assert(g.edges.count() === 3L)
  }

  test("toGraphFrame and fromGraphFrame round-trip") {
    val g = GraphDataset(vertices, edges)
    val gf = g.toGraphFrame
    assert(gf.vertices.columns.toSet === Set("id", "attr"))
    assert(gf.edges.columns.toSet === Set("src", "dst", "attr"))

    val g2 = GraphDataset.fromGraphFrame[String, String](gf)
    assert(g2.vertices.collect().toSet === g.vertices.collect().toSet)
    assert(g2.edges.collect().toSet === g.edges.collect().toSet)
  }

  test("triplets") {
    val g = GraphDataset(vertices, edges)
    val triplets = g.triplets.collect().toSet
    val expected = Set(
      GraphTriplet(1L, 2L, "A", "B", "love"),
      GraphTriplet(2L, 1L, "B", "A", "hate"),
      GraphTriplet(2L, 3L, "B", "C", "follow"))
    assert(triplets === expected)

    val t = GraphTriplet(1L, 2L, "A", "B", "love")
    assert(t.src === GraphVertex(1L, "A"))
    assert(t.dst === GraphVertex(2L, "B"))
    assert(t.edge === GraphEdge(1L, 2L, "love"))
    assert(t.vertexAttr(1L) === "A")
    assert(t.vertexAttr(2L) === "B")
    assert(t.otherVertexAttr(1L) === "B")
    assert(t.otherVertexAttr(2L) === "A")
    val expectedTuple = ((1L, "A"), (2L, "B"), "love")
    assert(t.toTuple === expectedTuple)
    val et = t.toEdgeTriplet
    assert(et.srcId === 1L && et.dstId === 2L && et.srcAttr === "A" && et.dstAttr === "B")
    assert(et.attr === "love")
  }

  test("filterVertices removes dropped vertices and incident edges") {
    val g = GraphDataset(vertices, edges)
    val filtered = g.filterVertices(v => v.id != 3L)
    assert(filtered.vertices.collect().map(_.id).toSet === Set(1L, 2L))
    assert(
      filtered.edges.collect().toSet === Set(
        GraphEdge(1L, 2L, "love"),
        GraphEdge(2L, 1L, "hate")))
  }

  test("filterEdges keeps all vertices") {
    val g = GraphDataset(vertices, edges)
    val filtered = g.filterEdges(e => e.attr == "love")
    assert(filtered.vertices.collect().map(_.id).toSet === Set(1L, 2L, 3L))
    assert(filtered.edges.collect().toSet === Set(GraphEdge(1L, 2L, "love")))
  }

  test("subgraph") {
    val g = GraphDataset(vertices, edges)
    val sub = g.subgraph(epred = e => e.attr != "hate", vpred = v => v.id <= 2L)
    assert(sub.vertices.collect().map(_.id).toSet === Set(1L, 2L))
    assert(sub.edges.collect().toSet === Set(GraphEdge(1L, 2L, "love")))
  }

  test("dropIsolatedVertices") {
    import sqlImplicits.*
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, "A"),
        GraphVertex(2L, "B"),
        GraphVertex(3L, "C"),
        GraphVertex(4L, "Isolated")))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, "love")))
    val g = GraphDataset(vv, ee)
    val dropped = g.dropIsolatedVertices()
    assert(dropped.vertices.collect().map(_.id).toSet === Set(1L, 2L))
    assert(dropped.edges.collect().toSet === Set(GraphEdge(1L, 2L, "love")))
  }

  test("mapVertices, mapEdges, mapTriplets") {
    val g = GraphDataset(vertices, edges)
    val mv = g.mapVertices((id, attr) => s"$id:$attr")
    assert(
      mv.vertices.collect().toSet === Set(
        GraphVertex(1L, "1:A"),
        GraphVertex(2L, "2:B"),
        GraphVertex(3L, "3:C")))

    val me = g.mapEdges(e => e.attr.length)
    assert(
      me.edges.collect().toSet === Set(
        GraphEdge(1L, 2L, 4),
        GraphEdge(2L, 1L, 4),
        GraphEdge(2L, 3L, 6)))

    val mt = g.mapTriplets(t => s"${t.srcAttr}-${t.attr}->${t.dstAttr}")
    assert(
      mt.edges.collect().toSet === Set(
        GraphEdge(1L, 2L, "A-love->B"),
        GraphEdge(2L, 1L, "B-hate->A"),
        GraphEdge(2L, 3L, "B-follow->C")))
  }

  test("asReversed and asUndirected") {
    import sqlImplicits.*
    val vv = spark.createDataset(Seq(GraphVertex(1L, "A"), GraphVertex(2L, "B")))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, "knows")))
    val g = GraphDataset(vv, ee)

    assert(g.asReversed().edges.collect().toSet === Set(GraphEdge(2L, 1L, "knows")))
    assert(
      g.asUndirected().edges.collect().toSet === Set(
        GraphEdge(1L, 2L, "knows"),
        GraphEdge(2L, 1L, "knows")))
  }

  test("validate") {
    import sqlImplicits.*
    val g = GraphDataset(vertices, edges)
    g.validate()

    val dupVertices = spark.createDataset(Seq(GraphVertex(1L, "A"), GraphVertex(1L, "A2")))
    val gDup = GraphDataset(dupVertices, edges)
    intercept[InvalidGraphException] {
      gDup.validate()
    }

    val missingVertexEdges = spark.createDataset(Seq(GraphEdge(1L, 99L, "broken")))
    val gMissing = GraphDataset(vertices, missingVertexEdges)
    intercept[InvalidGraphException] {
      gMissing.validate()
    }
    // checkVertices = false should only check duplicate vertices
    gMissing.validate(checkVertices = false, StorageLevel.MEMORY_ONLY)
  }

  test("custom case class attributes across DataFrame/GraphFrame boundaries") {
    import sqlImplicits.*
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, PersonAttr("Alice", 30)),
        GraphVertex(2L, PersonAttr("Bob", 25)),
        GraphVertex(3L, PersonAttr("Charlie", 40))))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, RelationAttr("friend", 0.9))))
    val g = GraphDataset(vv, ee)

    // triplets (DataFrame join -> .as[GraphTriplet[PersonAttr, RelationAttr]])
    val triplets = g.triplets.collect()
    assert(triplets.length === 1)
    assert(triplets.head.srcAttr === PersonAttr("Alice", 30))
    assert(triplets.head.dstAttr === PersonAttr("Bob", 25))
    assert(triplets.head.attr === RelationAttr("friend", 0.9))

    // toGraphFrame -> fromGraphFrame round-trip with struct attr columns
    val roundTripped = GraphDataset.fromGraphFrame[PersonAttr, RelationAttr](g.toGraphFrame)
    assert(roundTripped.vertices.collect().toSet === vv.collect().toSet)
    assert(roundTripped.edges.collect().toSet === ee.collect().toSet)

    // filterVertices (DataFrame left_semi join -> .as[GraphEdge[RelationAttr]])
    val filtered = g.filterVertices(_.attr.age <= 30)
    assert(
      filtered.vertices.collect().toSet === Set(
        GraphVertex(1L, PersonAttr("Alice", 30)),
        GraphVertex(2L, PersonAttr("Bob", 25))))
    assert(filtered.edges.collect().toSet === Set(GraphEdge(1L, 2L, RelationAttr("friend", 0.9))))

    // dropIsolatedVertices (DataFrame left_semi join -> .as[GraphVertex[PersonAttr]])
    val connected = g.dropIsolatedVertices()
    assert(
      connected.vertices.collect().toSet === Set(
        GraphVertex(1L, PersonAttr("Alice", 30)),
        GraphVertex(2L, PersonAttr("Bob", 25))))

    // connectedComponents (wraps GraphFrame.connectedComponents with struct attrs)
    val cc = g.connectedComponents.run().collect().map(r => r.id -> r.connectedComponent).toMap
    assert(cc === Map(1L -> 1L, 2L -> 1L, 3L -> 3L))
  }

  test("toGraphX conversion") {
    val g = GraphDataset(vertices, edges)
    val gx = g.toGraphX
    assert(gx.vertices.collect().toMap === localVertices)
    val gxEdges = gx.edges.collect().map(e => (e.srcId, e.dstId) -> e.attr).toMap
    assert(gxEdges === localEdges)
  }

  test("empty graph operations") {
    import sqlImplicits.*
    val emptyV = spark.emptyDataset[GraphVertex[String]]
    val emptyE = spark.emptyDataset[GraphEdge[String]]
    val g = GraphDataset(emptyV, emptyE)

    g.validate()
    assert(g.vertices.isEmpty)
    assert(g.edges.isEmpty)
    assert(g.triplets.isEmpty)
    assert(g.degrees.isEmpty)
    assert(g.inDegrees.isEmpty)
    assert(g.outDegrees.isEmpty)
    assert(g.dropIsolatedVertices().vertices.isEmpty)
  }

  test("validate preserves pre-cached caller vertices and edges") {
    import sqlImplicits.*
    val vv = spark.createDataset(Seq(GraphVertex(1L, "A"), GraphVertex(2L, "B"))).cache()
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, "know"))).cache()
    val _ = (vv.count(), ee.count())
    val g = GraphDataset(vv, ee)
    try {
      g.validate()
      assert(vv.storageLevel !== StorageLevel.NONE)
      assert(ee.storageLevel !== StorageLevel.NONE)
    } finally {
      vv.unpersist(blocking = true)
      ee.unpersist(blocking = true)
    }
  }

  test("fromGraphFrame rejects non-integral IDs and missing attr columns") {
    val validVertices = spark.createDataFrame(Seq((1L, "A"), (2L, "B"))).toDF("id", "attr")
    val validEdges = spark.createDataFrame(Seq((1L, 2L, "know"))).toDF("src", "dst", "attr")

    val strV = spark.createDataFrame(Seq(("1", "A"))).toDF("id", "attr")
    intercept[IllegalArgumentException] {
      val _ = GraphDataset.fromGraphFrame[String, String](GraphFrame(strV, validEdges))
    }

    val dblE = spark.createDataFrame(Seq((1.0, 2.0, "know"))).toDF("src", "dst", "attr")
    intercept[IllegalArgumentException] {
      val _ = GraphDataset.fromGraphFrame[String, String](GraphFrame(validVertices, dblE))
    }

    val noAttrV = spark.createDataFrame(Seq((1L, "A"))).toDF("id", "name")
    intercept[IllegalArgumentException] {
      val _ = GraphDataset.fromGraphFrame[String, String](GraphFrame(noAttrV, validEdges))
    }

    val noAttrE = spark.createDataFrame(Seq((1L, 2L, "know"))).toDF("src", "dst", "rel")
    intercept[IllegalArgumentException] {
      val _ = GraphDataset.fromGraphFrame[String, String](GraphFrame(validVertices, noAttrE))
    }
  }

  test("fromEdges supports explicit [VD, ED] type parameters") {
    val g = GraphDataset.fromEdges[String, String](edges, "default")
    assert(g.vertices.count() === 3L)
    assert(g.edges.count() === 3L)
  }

  test("asReversed and asUndirected with custom case class ED and self-loop") {
    import sqlImplicits.*
    val vv = spark.createDataset(
      Seq(GraphVertex(1L, PersonAttr("A", 1)), GraphVertex(2L, PersonAttr("B", 2))))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(1L, 2L, RelationAttr("friend", 1.0)),
        GraphEdge(1L, 1L, RelationAttr("self", 0.5))))
    val g = GraphDataset(vv, ee)

    assert(
      g.asReversed().edges.collect().toSet === Set(
        GraphEdge(2L, 1L, RelationAttr("friend", 1.0)),
        GraphEdge(1L, 1L, RelationAttr("self", 0.5))))
    // Self-loop is duplicated when both forward and reversed edges are unioned
    assert(g.asUndirected().edges.count() === 4L)
  }
}
