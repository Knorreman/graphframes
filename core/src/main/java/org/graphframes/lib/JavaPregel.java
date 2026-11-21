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
import org.apache.spark.api.java.function.Function3;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.ReduceFunction;
import org.apache.spark.graphframes.graphx.EdgeDirection;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.JavaGraphDataset;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple5;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Typed Java Pregel bulk-synchronous message-passing API for {@link JavaGraphDataset}.
 *
 * @param <VD> the vertex attribute type
 * @param <ED> the edge attribute type
 */
public final class JavaPregel<VD, ED> implements Serializable {

  private final JavaGraphDataset<VD, ED> graph;
  private int maxIter = 10;
  private int checkpointInterval = 2;
  private EdgeDirection activeDirection = EdgeDirection.Either();
  private boolean useLocalCheckpoints = false;
  private StorageLevel intermediateStorageLevel = StorageLevel.MEMORY_AND_DISK();

  public JavaPregel(JavaGraphDataset<VD, ED> graph) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
  }

  public JavaPregel<VD, ED> setMaxIter(int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(
          "Maximum number of iterations must be greater than 0, but got " + value);
    }
    this.maxIter = value;
    return this;
  }

  public JavaPregel<VD, ED> setCheckpointInterval(int value) {
    this.checkpointInterval = value;
    return this;
  }

  public JavaPregel<VD, ED> setActiveDirection(EdgeDirection value) {
    this.activeDirection = Objects.requireNonNull(value, "activeDirection must not be null");
    return this;
  }

  public JavaPregel<VD, ED> setUseLocalCheckpoints(boolean value) {
    this.useLocalCheckpoints = value;
    return this;
  }

  public JavaPregel<VD, ED> setIntermediateStorageLevel(StorageLevel value) {
    this.intermediateStorageLevel =
        Objects.requireNonNull(value, "intermediateStorageLevel must not be null");
    return this;
  }

  /** Starts a typed Pregel builder for message type {@code A}. */
  public <A> Builder<VD, ED, A> withMessages(A initialMsg, Encoder<A> msgEncoder) {
    return new Builder<>(
        graph,
        initialMsg,
        msgEncoder,
        maxIter,
        checkpointInterval,
        activeDirection,
        useLocalCheckpoints,
        intermediateStorageLevel);
  }

  /**
   * Fluent typed Pregel builder for message type {@code A}.
   */
  public static final class Builder<VD, ED, A> implements Serializable {
    private final JavaGraphDataset<VD, ED> graph;
    private final A initialMsg;
    private final Encoder<A> msgEncoder;
    private final int maxIterations;
    private final int checkpointInterval;
    private final EdgeDirection activeDirection;
    private final boolean useLocalCheckpoints;
    private final StorageLevel storageLevel;

    private Function3<Long, VD, A, VD> vertexProgram;
    private ReduceFunction<A> messageCombiner;
    private final List<FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, A>>> senders =
        new ArrayList<>();

    Builder(
        JavaGraphDataset<VD, ED> graph,
        A initialMsg,
        Encoder<A> msgEncoder,
        int maxIterations,
        int checkpointInterval,
        EdgeDirection activeDirection,
        boolean useLocalCheckpoints,
        StorageLevel storageLevel) {
      this.graph = graph;
      this.initialMsg = initialMsg;
      this.msgEncoder = Objects.requireNonNull(msgEncoder, "msgEncoder must not be null");
      this.maxIterations = maxIterations;
      this.checkpointInterval = checkpointInterval;
      this.activeDirection = activeDirection;
      this.useLocalCheckpoints = useLocalCheckpoints;
      this.storageLevel = storageLevel;
    }

    /** Sets the vertex program {@code (vertexId, currentAttr, aggregatedMsg) -> newAttr}. */
    public Builder<VD, ED, A> vprog(Function3<Long, VD, A, VD> fn) {
      this.vertexProgram = fn;
      return this;
    }

    /** Sends a message to the destination vertex of each active triplet. */
    public Builder<VD, ED, A> sendMsgToDst(MapFunction<GraphTriplet<VD, ED>, A> fn) {
      senders.add(t -> Collections.singletonList(new Tuple2<>(t.dstId(), fn.call(t))).iterator());
      return this;
    }

    /** Conditionally sends a message to the destination vertex of each active triplet. */
    public Builder<VD, ED, A> sendMsgToDstOpt(MapFunction<GraphTriplet<VD, ED>, Optional<A>> fn) {
      senders.add(t -> {
        Optional<A> opt = fn.call(t);
        return opt.map(m -> Collections.singletonList(new Tuple2<>(t.dstId(), m)).iterator())
            .orElseGet(Collections::emptyIterator);
      });
      return this;
    }

    /** Sends a message to the source vertex of each active triplet. */
    public Builder<VD, ED, A> sendMsgToSrc(MapFunction<GraphTriplet<VD, ED>, A> fn) {
      senders.add(t -> Collections.singletonList(new Tuple2<>(t.srcId(), fn.call(t))).iterator());
      return this;
    }

    /** Conditionally sends a message to the source vertex of each active triplet. */
    public Builder<VD, ED, A> sendMsgToSrcOpt(MapFunction<GraphTriplet<VD, ED>, Optional<A>> fn) {
      senders.add(t -> {
        Optional<A> opt = fn.call(t);
        return opt.map(m -> Collections.singletonList(new Tuple2<>(t.srcId(), m)).iterator())
            .orElseGet(Collections::emptyIterator);
      });
      return this;
    }

    /** Sends zero or more {@code (targetVertexId, message)} pairs for each active triplet. */
    public Builder<VD, ED, A> sendMsg(FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, A>> fn) {
      senders.add(fn);
      return this;
    }

    /** Sets the associative, commutative function merging two messages for the same vertex. */
    public Builder<VD, ED, A> mergeMsg(ReduceFunction<A> fn) {
      this.messageCombiner = fn;
      return this;
    }

    /** Executes the Pregel computation and returns the updated {@link JavaGraphDataset}. */
    public JavaGraphDataset<VD, ED> run() throws IOException {
      if (vertexProgram == null) {
        throw new IllegalArgumentException("vprog must be set before calling run().");
      }
      if (messageCombiner == null) {
        throw new IllegalArgumentException("mergeMsg must be set before calling run().");
      }
      if (senders.isEmpty()) {
        throw new IllegalArgumentException(
            "At least one message sender must be set before calling run().");
      }

      SparkSession spark = graph.spark();
      boolean shouldCheckpoint = checkpointInterval > 0;
      if (shouldCheckpoint && !useLocalCheckpoints
          && spark.sparkContext().getCheckpointDir().isEmpty()) {
        scala.Option<String> confDir = spark.conf().getOption("spark.checkpoint.dir");
        if (confDir.isDefined()) {
          spark.sparkContext().setCheckpointDir(confDir.get());
        } else {
          throw new IOException(
              "Checkpoint directory is not set. Please set it first using sc.setCheckpointDir() "
                  + "or enable local checkpoints via setUseLocalCheckpoints(true).");
        }
      }

      Encoder<VD> vdEnc = graph.vertexAttrEncoder();
      Encoder<ED> edEnc = graph.edgeAttrEncoder();
      Encoder<Tuple3<Long, VD, Boolean>> activeVertexEnc =
          Encoders.tuple(Encoders.LONG(), vdEnc, Encoders.BOOLEAN());
      Encoder<Tuple2<Long, A>> msgPairEnc = Encoders.tuple(Encoders.LONG(), msgEncoder);
      Encoder<Tuple5<Long, Long, VD, VD, ED>> tripletEnc = graph.tripletEncoder();

      Function3<Long, VD, A, VD> vp = vertexProgram;
      ReduceFunction<A> mm = messageCombiner;
      A initMsg = initialMsg;
      List<FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, A>>> senderFns =
          new ArrayList<>(senders);

      Dataset<Tuple3<Long, Long, ED>> cachedEdges = graph.edges().persist(storageLevel);
      Dataset<Tuple3<Long, VD, Boolean>> currentVertices = graph.vertices()
          .map(
              (MapFunction<Tuple2<Long, VD>, Tuple3<Long, VD, Boolean>>)
                  v -> new Tuple3<>(v._1(), vp.call(v._1(), v._2(), initMsg), true),
              activeVertexEnc)
          .persist(storageLevel);

      Dataset<Tuple2<Long, A>> messages = computeMessages(
          buildActiveTriplets(currentVertices, cachedEdges, false, tripletEnc),
          senderFns,
          mm,
          msgPairEnc,
          msgEncoder)
          .persist(storageLevel);

      boolean hasMessages = !messages.isEmpty();
      int iteration = 0;

      try {
        while (hasMessages && iteration < maxIterations) {
          Dataset<Tuple3<Long, VD, Boolean>> prevVertices = currentVertices;
          Dataset<Tuple2<Long, A>> prevMessages = messages;

          Dataset<Tuple3<Long, VD, Boolean>> updated = prevVertices
              .joinWith(prevMessages, prevVertices.col("_1").equalTo(prevMessages.col("key")),
                  "left_outer")
              .map(
                  (MapFunction<Tuple2<Tuple3<Long, VD, Boolean>, Tuple2<Long, A>>,
                      Tuple3<Long, VD, Boolean>>) pair -> {
                    Tuple3<Long, VD, Boolean> v = pair._1();
                    Tuple2<Long, A> m = pair._2();
                    if (m == null) {
                      return new Tuple3<>(v._1(), v._2(), false);
                    }
                    return new Tuple3<>(v._1(), vp.call(v._1(), v._2(), m._2()), true);
                  },
                  activeVertexEnc);

          Dataset<Tuple3<Long, VD, Boolean>> maybeCheckpointed;
          if (shouldCheckpoint && (iteration + 1) % checkpointInterval == 0) {
            maybeCheckpointed = useLocalCheckpoints
                ? updated.localCheckpoint(true)
                : updated.checkpoint(true);
          } else {
            maybeCheckpointed = updated;
          }

          currentVertices = maybeCheckpointed.persist(storageLevel);

          if (iteration + 1 < maxIterations) {
            messages = computeMessages(
                buildActiveTriplets(currentVertices, cachedEdges, true, tripletEnc),
                senderFns,
                mm,
                msgPairEnc,
                msgEncoder)
                .persist(storageLevel);
            hasMessages = !messages.isEmpty();
          } else {
            currentVertices.isEmpty();
            hasMessages = false;
          }

          prevMessages.unpersist();
          prevVertices.unpersist();
          iteration++;
        }
      } finally {
        messages.unpersist();
        cachedEdges.unpersist();
      }

      Dataset<Tuple2<Long, VD>> finalVertices = currentVertices
          .map(
              (MapFunction<Tuple3<Long, VD, Boolean>, Tuple2<Long, VD>>)
                  v -> new Tuple2<>(v._1(), v._2()),
              graph.vertexEncoder())
          .persist(storageLevel);
      finalVertices.count();
      currentVertices.unpersist();
      return JavaGraphDataset.of(finalVertices, graph.edges(), vdEnc, edEnc);
    }

    private Dataset<Tuple5<Long, Long, VD, VD, ED>> buildActiveTriplets(
        Dataset<Tuple3<Long, VD, Boolean>> vv,
        Dataset<Tuple3<Long, Long, ED>> ee,
        boolean filterActive,
        Encoder<Tuple5<Long, Long, VD, VD, ED>> tripletEnc) {
      Dataset<Tuple3<Long, VD, Boolean>> srcBase = (filterActive
          && (activeDirection.equals(EdgeDirection.Out())
              || activeDirection.equals(EdgeDirection.Both())))
          ? vv.filter(functions.col("_3"))
          : vv;
      Dataset<Tuple3<Long, VD, Boolean>> dstBase = (filterActive
          && (activeDirection.equals(EdgeDirection.In())
              || activeDirection.equals(EdgeDirection.Both())))
          ? vv.filter(functions.col("_3"))
          : vv;

      Dataset<Row> srcV = srcBase.toDF("srcId", "srcAttr", "srcActive");
      Dataset<Row> dstV = dstBase.toDF("dstId", "dstAttr", "dstActive");

      Dataset<Row> joined = ee.toDF("srcId", "dstId", "attr")
          .join(srcV, "srcId")
          .join(dstV, "dstId");

      if (filterActive && activeDirection.equals(EdgeDirection.Either())) {
        joined = joined.filter(functions.col("srcActive").or(functions.col("dstActive")));
      }

      return joined
          .select(
              functions.col("srcId").alias("_1"),
              functions.col("dstId").alias("_2"),
              functions.col("srcAttr").alias("_3"),
              functions.col("dstAttr").alias("_4"),
              functions.col("attr").alias("_5"))
          .as(tripletEnc);
    }

    private Dataset<Tuple2<Long, A>> computeMessages(
        Dataset<Tuple5<Long, Long, VD, VD, ED>> triplets,
        List<FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, A>>> senderFns,
        ReduceFunction<A> mergeMsg,
        Encoder<Tuple2<Long, A>> msgPairEnc,
        Encoder<A> msgEnc) {
      return triplets
          .flatMap(
              (FlatMapFunction<Tuple5<Long, Long, VD, VD, ED>, Tuple2<Long, A>>) raw -> {
                GraphTriplet<VD, ED> triplet =
                    new GraphTriplet<>(raw._1(), raw._2(), raw._3(), raw._4(), raw._5());
                List<Tuple2<Long, A>> out = new ArrayList<>();
                for (FlatMapFunction<GraphTriplet<VD, ED>, Tuple2<Long, A>> fn : senderFns) {
                  Iterator<Tuple2<Long, A>> it = fn.call(triplet);
                  while (it.hasNext()) {
                    out.add(it.next());
                  }
                }
                return out.iterator();
              },
              msgPairEnc)
          .groupByKey((MapFunction<Tuple2<Long, A>, Long>) Tuple2::_1, Encoders.LONG())
          .mapValues((MapFunction<Tuple2<Long, A>, A>) Tuple2::_2, msgEnc)
          .reduceGroups(mergeMsg);
    }
  }
}
