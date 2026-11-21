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

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithMaxIter
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.VertexId
import org.graphframes.lib.SVDPlusPlus

import java.util

/**
 * Result row for [[SVDPlusPlusDataset]], containing learned explicit factors (`column1`),
 * implicit factors (`column2`), bias (`column3`), and normalization term (`column4`).
 */
case class SVDPlusPlusResult(
    id: VertexId,
    column1: Seq[Double],
    column2: Seq[Double],
    column3: Double,
    column4: Double)
    extends Serializable {

  /** Returns `column1` as an unmodifiable Java {@code List<Double>}. */
  def column1AsJava: util.List[java.lang.Double] = toJavaDoubleList(column1)

  /** Returns `column2` as an unmodifiable Java {@code List<Double>}. */
  def column2AsJava: util.List[java.lang.Double] = toJavaDoubleList(column2)

  private def toJavaDoubleList(values: Seq[Double]): util.List[java.lang.Double] = {
    val list = new util.ArrayList[java.lang.Double](values.size)
    values.foreach { d =>
      val _ = list.add(java.lang.Double.valueOf(d))
    }
    util.Collections.unmodifiableList(list)
  }
}

object SVDPlusPlusResult {
  implicit val encoder: Encoder[SVDPlusPlusResult] =
    Encoders.product[SVDPlusPlusResult]
}

private object SVDPlusPlusDatasetCols {
  val ID_COL = "id"
  val COLUMN1_COL = "column1"
  val COLUMN2_COL = "column2"
  val COLUMN3_COL = "column3"
  val COLUMN4_COL = "column4"
}

/**
 * SVD++ collaborative filtering algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.SVDPlusPlus]].
 *
 * Accepts a typed edge weight/rating extractor `ED => Double` (or uses an implicit `Numeric[ED]`
 * when `ED` is already numeric) instead of requiring a `"weight"` column name.
 */
class SVDPlusPlusDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with WithMaxIter
    with Logging {

  private var _rank: Int = 10
  private var _minVal: Double = 0.0
  private var _maxVal: Double = 5.0
  private var _gamma1: Double = 0.007
  private var _gamma2: Double = 0.007
  private var _gamma6: Double = 0.005
  private var _gamma7: Double = 0.015
  private var _loss: Option[Double] = None

  def rank(value: Int): this.type = {
    _rank = value
    this
  }

  def minValue(value: Double): this.type = {
    _minVal = value
    this
  }

  def maxValue(value: Double): this.type = {
    _maxVal = value
    this
  }

  def gamma1(value: Double): this.type = {
    _gamma1 = value
    this
  }

  def gamma2(value: Double): this.type = {
    _gamma2 = value
    this
  }

  def gamma6(value: Double): this.type = {
    _gamma6 = value
    this
  }

  def gamma7(value: Double): this.type = {
    _gamma7 = value
    this
  }

  /**
   * Runs SVD++ using `weightExtractor` to extract a rating from each edge attribute `ED`,
   * returning `(vertexEmbeddings, trainingLoss)` where `vertexEmbeddings` is persisted and
   * materialized. Callers should call `.unpersist()` on the returned dataset when finished.
   */
  def run(weightExtractor: ED => Double): (Dataset[SVDPlusPlusResult], Double) = {
    val session = graph.spark
    import session.implicits.*

    val vDF = graph.vertices.toDF(GraphFrame.ID, GraphFrame.ATTR)
    val eDF = graph.edges
      .map(e => (e.srcId, e.dstId, weightExtractor(e.attr)))
      .toDF(GraphFrame.SRC, GraphFrame.DST, SVDPlusPlus.COLUMN_WEIGHT)

    val svd = GraphFrame(vDF, eDF).svdPlusPlus
      .rank(_rank)
      .minValue(_minVal)
      .maxValue(_maxVal)
      .gamma1(_gamma1)
      .gamma2(_gamma2)
      .gamma6(_gamma6)
      .gamma7(_gamma7)
    maxIter.foreach(svd.maxIter)

    val resDF = svd.run()
    val lossVal = svd.loss
    _loss = Some(lossVal)

    import SVDPlusPlusDatasetCols.*
    val typed = resDF
      .select(
        col(GraphFrame.ID).alias(ID_COL),
        col(SVDPlusPlus.COLUMN1).alias(COLUMN1_COL),
        col(SVDPlusPlus.COLUMN2).alias(COLUMN2_COL),
        col(SVDPlusPlus.COLUMN3).alias(COLUMN3_COL),
        col(SVDPlusPlus.COLUMN4).alias(COLUMN4_COL))
      .as[SVDPlusPlusResult]

    val persisted =
      DatasetCacheUtils.materializeAndReleaseRaw(resDF, typed, StorageLevel.MEMORY_AND_DISK)
    resultIsPersistent()
    (persisted, lossVal)
  }

  /**
   * Convenience overload for graphs whose edge attribute `ED` is already `Numeric`.
   */
  def run()(implicit num: Numeric[ED]): (Dataset[SVDPlusPlusResult], Double) = {
    run((ed: ED) => num.toDouble(ed))
  }

  /** Returns the training loss from the most recent `run` call. */
  def loss: Double = {
    _loss.getOrElse(throw new IllegalStateException("The algorithm has not been run yet"))
  }
}
