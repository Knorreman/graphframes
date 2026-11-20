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
import org.apache.spark.graphframes.graphx.EdgeTriplet
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphDataset
import org.graphframes.JobDescription
import org.graphframes.Logging
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithJobDescriptionPrefix
import org.graphframes.WithLocalCheckpoints

import java.io.IOException
import scala.collection.mutable.ListBuffer
import scala.reflect.runtime.universe.TypeTag

private[lib] case class ActiveGraphVertex[VD](id: VertexId, attr: VD, active: Boolean)
    extends Serializable

private[lib] object ActiveGraphVertex {
  implicit def encoder[VD: TypeTag]: Encoder[ActiveGraphVertex[VD]] =
    Encoders.product[ActiveGraphVertex[VD]]
}

/**
 * Typed Pregel-like bulk-synchronous message-passing API for [[org.graphframes.GraphDataset]].
 *
 * You can construct an instance via [[org.graphframes.GraphDataset#pregel]], configure iteration
 * and checkpoint settings using the builder methods, and either:
 *   - call [[run]] or [[runTriplets]] directly with `vprog`, `sendMsg`, and `mergeMsg`, or
 *   - use the fluent typed builder via [[withMessages]] (`vprog`, `sendMsgToDst`, `sendMsgToSrc`,
 *     `mergeMsg`, `run()`).
 *
 * @tparam VD
 *   the vertex attribute type
 * @tparam ED
 *   the edge attribute type
 */
class PregelGraphDataset[VD: TypeTag, ED: TypeTag] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Logging
    with WithLocalCheckpoints
    with WithIntermediateStorageLevel
    with WithJobDescriptionPrefix {

  override protected def defaultJobDescriptionPrefix: String = "GraphFrames PregelGraphDataset"

  private var maxIter: Int = 10
  private var checkpointInterval: Int = 2
  private var activeDirection: EdgeDirection = EdgeDirection.Either

  /** Sets the max number of iterations (default: 10). */
  def setMaxIter(value: Int): this.type = {
    require(value > 0, s"Maximum number of iterations must be greater than 0, but got $value.")
    maxIter = value
    this
  }

  /** Gets the max number of iterations. */
  def getMaxIter: Int = maxIter

  /**
   * Sets the number of iterations between two checkpoints (default: 2).
   *
   * Checkpoint is disabled if this is set to 0.
   */
  def setCheckpointInterval(value: Int): this.type = {
    checkpointInterval = value
    this
  }

  /** Gets the checkpoint interval. */
  def getCheckpointInterval: Int = checkpointInterval

  /**
   * Sets the direction of edges incident to active vertices on which `sendMsg` is evaluated
   * (default: `EdgeDirection.Either`).
   */
  def setActiveDirection(value: EdgeDirection): this.type = {
    activeDirection = value
    this
  }

  /** Gets the active edge direction. */
  def getActiveDirection: EdgeDirection = activeDirection

  /**
   * Starts a fluent typed Pregel builder with message type `A` and initial message `initialMsg`.
   *
   * @example
   *   {{{
   *   val result = graph.pregel
   *     .setMaxIter(10)
   *     .withMessages[Int](0)
   *     .vprog((_, attr, msg) => math.max(attr, msg))
   *     .sendMsgToDstOpt(t => if (t.srcAttr > t.dstAttr) Some(t.srcAttr) else None)
   *     .mergeMsg(math.max)
   *     .run()
   *   }}}
   */
  def withMessages[A: Encoder](initialMsg: A): TypedPregelBuilder[VD, ED, A] = {
    new TypedPregelBuilder[VD, ED, A](
      graph = graph,
      initialMsg = initialMsg,
      maxIterations = maxIter,
      checkpointInterval = checkpointInterval,
      activeDirection = activeDirection,
      useLocalCheckpoints = useLocalCheckpoints,
      intermediateStorageLevel = intermediateStorageLevel,
      jobDescriptionPrefix = getJobDescriptionPrefix)
  }

  /**
   * Executes the Pregel computation using GraphX `EdgeTriplet` in `sendMsg`.
   */
  def run[A: Encoder](initialMsg: A)(
      vprog: (VertexId, VD, A) => VD,
      sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)],
      mergeMsg: (A, A) => A): GraphDataset[VD, ED] = {
    PregelGraphDataset.run(
      graph = graph,
      initialMsg = initialMsg,
      maxIterations = maxIter,
      checkpointInterval = checkpointInterval,
      activeDirection = activeDirection,
      useLocalCheckpoints = useLocalCheckpoints,
      intermediateStorageLevel = intermediateStorageLevel,
      jobDescriptionPrefix = getJobDescriptionPrefix,
      vprog = vprog,
      sendMsg = sendMsg,
      mergeMsg = mergeMsg)
  }

  /**
   * Executes the Pregel computation using typed [[GraphTriplet]] in `sendMsg`.
   */
  def runTriplets[A: Encoder](initialMsg: A)(
      vprog: (VertexId, VD, A) => VD,
      sendMsg: GraphTriplet[VD, ED] => Iterator[(VertexId, A)],
      mergeMsg: (A, A) => A): GraphDataset[VD, ED] = {
    PregelGraphDataset.runTriplets(
      graph = graph,
      initialMsg = initialMsg,
      maxIterations = maxIter,
      checkpointInterval = checkpointInterval,
      activeDirection = activeDirection,
      useLocalCheckpoints = useLocalCheckpoints,
      intermediateStorageLevel = intermediateStorageLevel,
      jobDescriptionPrefix = getJobDescriptionPrefix,
      vprog = vprog,
      sendMsg = sendMsg,
      mergeMsg = mergeMsg)
  }
}

/**
 * Fluent typed builder for configuring and running a [[PregelGraphDataset]] computation with a
 * fixed message type `A`.
 */
class TypedPregelBuilder[VD: TypeTag, ED: TypeTag, A: Encoder] private[lib] (
    private val graph: GraphDataset[VD, ED],
    private val initialMsg: A,
    private val maxIterations: Int,
    private val checkpointInterval: Int,
    private val activeDirection: EdgeDirection,
    private val useLocalCheckpoints: Boolean,
    private val intermediateStorageLevel: StorageLevel,
    private val jobDescriptionPrefix: String)
    extends Serializable {

  private var vertexProgram: Option[(VertexId, VD, A) => VD] = None
  private var messageCombiner: Option[(A, A) => A] = None
  private val senders = ListBuffer.empty[GraphTriplet[VD, ED] => Iterator[(VertexId, A)]]

  /** Sets the vertex program `(vertexId, currentAttr, aggregatedMsg) => newAttr`. */
  def vprog(f: (VertexId, VD, A) => VD): this.type = {
    vertexProgram = Some(f)
    this
  }

  /** Sends a message to the destination vertex of each active triplet. */
  def sendMsgToDst(f: GraphTriplet[VD, ED] => A): this.type = {
    senders += (t => Iterator.single((t.dstId, f(t))))
    this
  }

  /** Conditionally sends a message to the destination vertex of each active triplet. */
  def sendMsgToDstOpt(f: GraphTriplet[VD, ED] => Option[A]): this.type = {
    senders += (t => f(t).iterator.map(m => (t.dstId, m)))
    this
  }

  /** Sends a message to the source vertex of each active triplet. */
  def sendMsgToSrc(f: GraphTriplet[VD, ED] => A): this.type = {
    senders += (t => Iterator.single((t.srcId, f(t))))
    this
  }

  /** Conditionally sends a message to the source vertex of each active triplet. */
  def sendMsgToSrcOpt(f: GraphTriplet[VD, ED] => Option[A]): this.type = {
    senders += (t => f(t).iterator.map(m => (t.srcId, m)))
    this
  }

  /** Sends zero or more `(targetVertexId, message)` pairs for each active triplet. */
  def sendMsg(f: GraphTriplet[VD, ED] => Iterator[(VertexId, A)]): this.type = {
    senders += f
    this
  }

  /** Sets the commutative, associative function used to combine messages for the same vertex. */
  def mergeMsg(f: (A, A) => A): this.type = {
    messageCombiner = Some(f)
    this
  }

  /** Executes the configured Pregel computation. */
  def run(): GraphDataset[VD, ED] = {
    val vp = vertexProgram.getOrElse(
      throw new IllegalArgumentException("vprog must be set before calling run()."))
    val mm = messageCombiner.getOrElse(
      throw new IllegalArgumentException("mergeMsg must be set before calling run()."))
    require(
      senders.nonEmpty,
      "At least one message sender (sendMsgToDst, sendMsgToSrc, or sendMsg) must be set.")

    val senderFns = senders.toVector
    PregelGraphDataset.runTriplets(
      graph = graph,
      initialMsg = initialMsg,
      maxIterations = maxIterations,
      checkpointInterval = checkpointInterval,
      activeDirection = activeDirection,
      useLocalCheckpoints = useLocalCheckpoints,
      intermediateStorageLevel = intermediateStorageLevel,
      jobDescriptionPrefix = jobDescriptionPrefix,
      vprog = vp,
      sendMsg = t => senderFns.iterator.flatMap(fn => fn(t)),
      mergeMsg = mm)
  }
}

object PregelGraphDataset extends Logging {

  /**
   * Execute a typed Pregel-like iterative vertex-parallel computation on a [[GraphDataset]] using
   * GraphX `EdgeTriplet` in `sendMsg`.
   *
   * On the first iteration all vertices receive `initialMsg`. On subsequent iterations `vprog` is
   * invoked only on vertices that received at least one message. Execution continues until no
   * messages are sent or `maxIterations` iterations have completed.
   */
  def run[VD: TypeTag, ED: TypeTag, A: Encoder](
      graph: GraphDataset[VD, ED],
      initialMsg: A,
      maxIterations: Int = Int.MaxValue,
      checkpointInterval: Int = 0,
      activeDirection: EdgeDirection = EdgeDirection.Either,
      useLocalCheckpoints: Boolean = false,
      intermediateStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      jobDescriptionPrefix: String = "GraphFrames PregelGraphDataset",
      vprog: (VertexId, VD, A) => VD,
      sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)],
      mergeMsg: (A, A) => A): GraphDataset[VD, ED] = {
    runTriplets(
      graph = graph,
      initialMsg = initialMsg,
      maxIterations = maxIterations,
      checkpointInterval = checkpointInterval,
      activeDirection = activeDirection,
      useLocalCheckpoints = useLocalCheckpoints,
      intermediateStorageLevel = intermediateStorageLevel,
      jobDescriptionPrefix = jobDescriptionPrefix,
      vprog = vprog,
      sendMsg = (t: GraphTriplet[VD, ED]) => sendMsg(t.toEdgeTriplet),
      mergeMsg = mergeMsg)
  }

  /**
   * Execute a typed Pregel-like iterative vertex-parallel computation on a [[GraphDataset]] using
   * typed [[GraphTriplet]] in `sendMsg`.
   */
  def runTriplets[VD: TypeTag, ED: TypeTag, A: Encoder](
      graph: GraphDataset[VD, ED],
      initialMsg: A,
      maxIterations: Int = Int.MaxValue,
      checkpointInterval: Int = 0,
      activeDirection: EdgeDirection = EdgeDirection.Either,
      useLocalCheckpoints: Boolean = false,
      intermediateStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      jobDescriptionPrefix: String = "GraphFrames PregelGraphDataset",
      vprog: (VertexId, VD, A) => VD,
      sendMsg: GraphTriplet[VD, ED] => Iterator[(VertexId, A)],
      mergeMsg: (A, A) => A): GraphDataset[VD, ED] = {
    require(
      maxIterations > 0,
      s"Maximum number of iterations must be greater than 0, but got $maxIterations.")

    val spark = graph.spark
    val shouldCheckpoint = checkpointInterval > 0
    if (shouldCheckpoint && !useLocalCheckpoints && spark.sparkContext.getCheckpointDir.isEmpty) {
      spark.conf.getOption("spark.checkpoint.dir") match {
        case Some(d) => spark.sparkContext.setCheckpointDir(d)
        case None =>
          throw new IOException(
            "Checkpoint directory is not set. Please set it first using sc.setCheckpointDir() " +
              "or enable local checkpoints via setUseLocalCheckpoints(true).")
      }
    }

    val maxIterSuffix = if (maxIterations == Int.MaxValue) "" else s" / $maxIterations"

    JobDescription.withRestoredJobDescription(spark) {
      implicit val msgPairEncoder: Encoder[(VertexId, A)] =
        Encoders.tuple(Encoders.scalaLong, implicitly[Encoder[A]])
      implicit val vertexIdEncoder: Encoder[VertexId] = Encoders.scalaLong

      def buildActiveTriplets(
          vv: Dataset[ActiveGraphVertex[VD]],
          ee: Dataset[GraphEdge[ED]],
          filterActive: Boolean): Dataset[GraphTriplet[VD, ED]] = {
        val srcV = {
          val base =
            if (filterActive && (activeDirection == EdgeDirection.Out || activeDirection == EdgeDirection.Both)) {
              vv.filter(col("active"))
            } else {
              vv
            }
          base.toDF("srcId", "srcAttr", "srcActive")
        }
        val dstV = {
          val base =
            if (filterActive && (activeDirection == EdgeDirection.In || activeDirection == EdgeDirection.Both)) {
              vv.filter(col("active"))
            } else {
              vv
            }
          base.toDF("dstId", "dstAttr", "dstActive")
        }

        val joined = ee
          .toDF("srcId", "dstId", "attr")
          .join(srcV, Seq("srcId"), "inner")
          .join(dstV, Seq("dstId"), "inner")

        val filtered =
          if (filterActive && activeDirection == EdgeDirection.Either) {
            joined.filter(col("srcActive") || col("dstActive"))
          } else {
            joined
          }

        filtered
          .select(col("srcId"), col("dstId"), col("srcAttr"), col("dstAttr"), col("attr"))
          .as[GraphTriplet[VD, ED]]
      }

      def computeMessages(triplets: Dataset[GraphTriplet[VD, ED]]): Dataset[(VertexId, A)] = {
        triplets
          .flatMap(sendMsg)
          .groupByKey(_._1)
          .mapValues(_._2)
          .reduceGroups(mergeMsg)
      }

      spark.sparkContext.setJobDescription(s"$jobDescriptionPrefix: initialization")
      val cachedEdges = graph.edges.persist(intermediateStorageLevel)
      var currentVertices: Dataset[ActiveGraphVertex[VD]] = graph.vertices
        .map(v => ActiveGraphVertex(v.id, vprog(v.id, v.attr, initialMsg), active = true))
        .persist(intermediateStorageLevel)

      var messages: Dataset[(VertexId, A)] =
        computeMessages(buildActiveTriplets(currentVertices, cachedEdges, filterActive = false))
          .persist(intermediateStorageLevel)

      var hasMessages = !messages.isEmpty
      var iteration = 0

      try {
        while (hasMessages && iteration < maxIterations) {
          spark.sparkContext.setJobDescription(
            s"$jobDescriptionPrefix: iteration ${iteration + 1}$maxIterSuffix")
          val prevVertices = currentVertices
          val prevMessages = messages

          val updated: Dataset[ActiveGraphVertex[VD]] = prevVertices
            .joinWith(prevMessages, prevVertices("id") === prevMessages("key"), "left_outer")
            .map {
              case (v, null) => ActiveGraphVertex(v.id, v.attr, active = false)
              case (v, (_, msg)) =>
                ActiveGraphVertex(v.id, vprog(v.id, v.attr, msg), active = true)
            }

          val maybeCheckpointed =
            if (shouldCheckpoint && (iteration + 1) % checkpointInterval == 0) {
              if (useLocalCheckpoints) {
                updated.localCheckpoint(eager = true)
              } else {
                updated.checkpoint(eager = true)
              }
            } else {
              updated
            }

          currentVertices = maybeCheckpointed.persist(intermediateStorageLevel)

          if (iteration + 1 < maxIterations) {
            messages = computeMessages(
              buildActiveTriplets(currentVertices, cachedEdges, filterActive = true))
              .persist(intermediateStorageLevel)
            hasMessages = !messages.isEmpty
          } else {
            val _ = currentVertices.isEmpty
            hasMessages = false
          }

          prevMessages.unpersist()
          prevVertices.unpersist()
          iteration += 1
        }
      } finally {
        messages.unpersist()
        cachedEdges.unpersist()
        ()
      }

      spark.sparkContext.setJobDescription(s"$jobDescriptionPrefix: materializing final result")
      val finalVertices = currentVertices
        .map(v => GraphVertex(v.id, v.attr))
        .persist(intermediateStorageLevel)
      val _ = finalVertices.count()
      currentVertices.unpersist()
      GraphDataset(finalVertices, graph.edges)
    }
  }
}
