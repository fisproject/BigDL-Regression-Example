package com.intel.analytics.bigdl.example

import com.intel.analytics.bigdl._ // Module型が使えるようになる
import com.intel.analytics.bigdl.nn._

object SimpleRegression {
  import com.intel.analytics.bigdl.numeric.NumericFloat
  def apply(): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(4))) // feature size
    .add(Linear(4, 8).setName("func_1"))
    .add(ReLU(true))
    .add(Linear(8, 1).setName("func_2"))
    .add(Mean())
    model
  }
}
