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

package org.graphframes.dataset.lib

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.GraphVertexBase
import org.graphframes.dataset.VertexId
import org.graphframes.embeddings.Hash2Vec
import org.graphframes.embeddings.RandomWalkEmbeddings
import org.graphframes.rw.RandomWalkBase

/**
 * Result row for [[RandomWalkEmbeddingsDataset]], containing a vertex ID and embedding vector.
 */
case class NodeEmbeddingResult(id: VertexId, embedding: Seq[Double])
    extends GraphVertexBase[Seq[Double]] {
  def attr: Seq[Double] = embedding
  def toVertex: GraphVertex[Seq[Double]] = GraphVertex(id, embedding)
}

object NodeEmbeddingResult {
  implicit val encoder: Encoder[NodeEmbeddingResult] =
    Encoders.product[NodeEmbeddingResult]
}

private object RandomWalkEmbeddingsDatasetCols {
  val ID_COL = "id"
  val EMBEDDING_COL = "embedding"
}

/**
 * Random-walk-based node embeddings for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.embeddings.RandomWalkEmbeddings]].
 */
class RandomWalkEmbeddingsDataset[VD, ED] private[dataset] (
    private val graph: GraphDataset[VD, ED])
    extends Serializable
    with Logging
    with WithIntermediateStorageLevel {

  private var sequenceModel: Option[Either[Word2Vec, Hash2Vec]] = None
  private var randomWalks: Option[RandomWalkBase] = None
  private var aggregateNeighbors: Boolean = true
  private var useEdgeDirections: Boolean = false
  private var maxNbrs: Int = 50
  private var seed: Long = 42L
  private var cachedRwPath: Option[String] = None
  private var cleanUpAfterRun: Boolean = false

  /** Sets the sequence model (`Left(Word2Vec)` or `Right(Hash2Vec)`). */
  def setSequenceModel(value: Either[Word2Vec, Hash2Vec]): this.type = {
    require(value != null, "sequenceModel must not be null.")
    sequenceModel = Some(value)
    this
  }

  /** Sets the random walk generator. */
  def setRandomWalks(value: RandomWalkBase): this.type = {
    require(value != null, "randomWalks must not be null.")
    randomWalks = Some(value)
    this
  }

  /** Sets the random seed (default: 42L). */
  def setSeed(value: Long): this.type = {
    seed = value
    this
  }

  /** Sets whether to use edge directions (default: `false`). */
  def setUseEdgeDirections(value: Boolean): this.type = {
    useEdgeDirections = value
    this
  }

  /** Sets whether to aggregate neighbor embeddings via min-hash sampling (default: `true`). */
  def setAggregateNeighbors(value: Boolean): this.type = {
    aggregateNeighbors = value
    this
  }

  /** Sets the maximum number of neighbors sampled for aggregation (default: 50). */
  def setMaxNbrs(value: Int): this.type = {
    maxNbrs = value
    this
  }

  /** Sets a path to pre-computed cached random walks in Parquet format. */
  def useCachedRandomWalks(path: String): this.type = {
    cachedRwPath = Some(path)
    this
  }

  /** Sets whether to clean up temporary walk files after running (default: `false`). */
  def setCleanUpAfterRun(value: Boolean): this.type = {
    cleanUpAfterRun = value
    this
  }

  /**
   * Runs random-walk node embeddings and returns a persisted and materialized dataset of
   * [[NodeEmbeddingResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[NodeEmbeddingResult] = {
    require(sequenceModel.isDefined, "sequenceModel must be set before calling run().")
    require(randomWalks.isDefined, "randomWalks must be set before calling run().")

    val rw = graph.toGraphFrame.randomWalksBasedEmbedding
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setSeed(seed)
      .setUseEdgeDirections(useEdgeDirections)
      .setAggregateNeighbors(aggregateNeighbors)
      .setMaxNbrs(maxNbrs)
      .setCleanUpAfterRun(cleanUpAfterRun)
    sequenceModel.foreach(rw.setSequenceModel)
    randomWalks.foreach(rw.setRandomWalks)
    cachedRwPath.foreach(rw.useCachedRandomWalks)

    import RandomWalkEmbeddingsDatasetCols.*
    val rawDF = rw.run()
    val typed = rawDF
      .select(
        col(GraphFrame.ID).cast("long").alias(ID_COL),
        vector_to_array(col(RandomWalkEmbeddings.embeddingColName)).alias(EMBEDDING_COL))
      .as[NodeEmbeddingResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}
