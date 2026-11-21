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

package org.graphframes.lib;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.ReduceFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.expressions.Aggregator;
import org.graphframes.JavaGraphDataset;
import scala.Tuple2;
import scala.Tuple5;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Typed Java message aggregation builder for {@link JavaGraphDataset}.
 *
 * @param <VD> the vertex attribute type
 * @param <ED> the edge attribute type
 * @param <M>  the message type
 */
public final class JavaAggregateMessages<VD, ED, M> implements Serializable {

  private final JavaGraphDataset<VD, ED> graph;
  private final Encoder<M> msgEncoder;
  private final List<FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, M>>> senders =
      new ArrayList<>();

  public JavaAggregateMessages(JavaGraphDataset<VD, ED> graph, Encoder<M> msgEncoder) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
    this.msgEncoder = Objects.requireNonNull(msgEncoder, "msgEncoder must not be null");
  }

  /** Sends a message to the source vertex of each triplet. */
  public JavaAggregateMessages<VD, ED, M> sendToSrc(MapFunction<GraphTriplet<VD, ED>, M> fn) {
    senders.add(t -> Collections.singletonList(new Tuple2<>(t.srcId(), fn.call(t))).iterator());
    return this;
  }

  /** Conditionally sends a message to the source vertex of each triplet. */
  public JavaAggregateMessages<VD, ED, M> sendToSrcOpt(
      MapFunction<GraphTriplet<VD, ED>, Optional<M>> fn) {
    senders.add(t -> {
      Optional<M> opt = fn.call(t);
      return opt.map(m -> Collections.singletonList(new Tuple2<>(t.srcId(), m)).iterator())
          .orElseGet(Collections::emptyIterator);
    });
    return this;
  }

  /** Sends a message to the destination vertex of each triplet. */
  public JavaAggregateMessages<VD, ED, M> sendToDst(MapFunction<GraphTriplet<VD, ED>, M> fn) {
    senders.add(t -> Collections.singletonList(new Tuple2<>(t.dstId(), fn.call(t))).iterator());
    return this;
  }

  /** Conditionally sends a message to the destination vertex of each triplet. */
  public JavaAggregateMessages<VD, ED, M> sendToDstOpt(
      MapFunction<GraphTriplet<VD, ED>, Optional<M>> fn) {
    senders.add(t -> {
      Optional<M> opt = fn.call(t);
      return opt.map(m -> Collections.singletonList(new Tuple2<>(t.dstId(), m)).iterator())
          .orElseGet(Collections::emptyIterator);
    });
    return this;
  }

  /** Sends zero or more {@code (targetVertexId, message)} pairs for each triplet. */
  public JavaAggregateMessages<VD, ED, M> sendMsg(
      FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, M>> fn) {
    senders.add(fn);
    return this;
  }

  private Dataset<Tuple2<Long, M>> buildRawMessages() {
    if (senders.isEmpty()) {
      throw new IllegalArgumentException(
          "At least one message sender must be configured using sendToSrc, sendToDst, or sendMsg.");
    }
    List<FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, M>>> senderFns =
        new ArrayList<>(senders);
    Encoder<Tuple2<Long, M>> pairEnc = Encoders.tuple(Encoders.LONG(), msgEncoder);
    return graph.triplets().flatMap(
        (FlatMapFunction<Tuple5<Long, Long, VD, VD, ED>, Tuple2<Long, M>>) raw -> {
          GraphTriplet<VD, ED> triplet =
              new GraphTriplet<>(raw._1(), raw._2(), raw._3(), raw._4(), raw._5());
          List<Tuple2<Long, M>> out = new ArrayList<>();
          for (FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, M>> fn : senderFns) {
            Iterator<Tuple2<Long, M>> it = fn.call(triplet);
            while (it.hasNext()) {
              out.add(it.next());
            }
          }
          return out.iterator();
        },
        pairEnc);
  }

  /** Reduces messages sent to each vertex using {@code mergeMsg}. */
  public Dataset<Tuple2<Long, M>> agg(ReduceFunction<M> mergeMsg) {
    return buildRawMessages()
        .groupByKey((MapFunction<Tuple2<Long, M>, Long>) Tuple2::_1, Encoders.LONG())
        .mapValues((MapFunction<Tuple2<Long, M>, M>) Tuple2::_2, msgEncoder)
        .reduceGroups(mergeMsg);
  }

  /** Aggregates messages sent to each vertex using a typed Spark {@code Aggregator}. */
  public <R> Dataset<Tuple2<Long, R>> agg(Aggregator<M, ?, R> aggregator) {
    return buildRawMessages()
        .groupByKey((MapFunction<Tuple2<Long, M>, Long>) Tuple2::_1, Encoders.LONG())
        .mapValues((MapFunction<Tuple2<Long, M>, M>) Tuple2::_2, msgEncoder)
        .agg(aggregator.toColumn());
  }
}
