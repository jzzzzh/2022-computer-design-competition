<template>
  <div class="bgBlurred">
  <el-row :gutter="12">
    <el-col :span="8">
      <el-card shadow="hover" style="height: 500px">
        <template #header>
          <div class="card-header">
            <span><p class="animate__animated animate__flipInX" style="font-size: 30px; font-family: JYDJW">什么是智投</p></span>
          </div>
        </template>
        <el-image :src="url0" style="height: 150px"></el-image>
        <p class="cardWord">
          智投是一个致力于在纷繁的量化投资市场中给投资者一个简洁清晰投资建议的量化投资系统，通过LSTM和GA的技术结合，
          对现有绝大多数投资产品进行预测，降低投资难度，给投资者冷静清晰的投资决策。
        </p>
      </el-card>
    </el-col>
    <el-col :span="8">
      <el-card shadow="hover" style="height: 500px">
        <template #header>
          <div class="card-header">
            <span><p class="animate__animated animate__flipInX" style="font-size: 30px; font-family: JYDJW">什么是LSTM</p></span>
          </div>
        </template>
         <el-image :src="url1" style="height: 150px"></el-image>
        <p class="cardWord">Long Short Term Memory networks（以下简称LSTMs），一种特殊的RNN网络，该网络设计出来是为了解决长依赖问题。该网络由
          Hochreiter & Schmidhuber (1997)引入，并有许多人对其进行了改进和普及。他们的工作被用来解决了各种各样的问题，直到目前还被广泛应用。</p>

      </el-card>
    </el-col>
    <el-col :span="8">
      <el-card shadow="hover" style="height: 500px">
        <template #header>
          <div class="card-header">
            <span><p class="animate__animated animate__flipInX" style="font-size: 30px; font-family: JYDJW">什么是NSGA</p></span>
          </div>
        </template>
        <el-image :src="url2" style="height: 150px"></el-image>
        <p class="cardWord">1995年，Srinivas和Deb提出了非支配遗传（Non-dominated Sorting Genetic Algorithms，NSGA）算法。NSGA算法是以遗传算法为基础并基于Pareto最优概念得到的。
          NSGA算法与基本遗传算法的主要区别是其在进行选择操作之前对个体进行了快速非支配排序，增大了优秀个体被保留的概率。</p>
      </el-card>
    </el-col>
  </el-row>
  </div>
</template>

<script>
export default {
  name: "MyIntro",
  data(){
    return{
      url0:"https://img1.baidu.com/it/u=774209527,2764708230&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=286",
      url1:"https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.pianshen.com%2Fimages%2F868%2F270334da78bc2b42ded7c9b4b8444ba4.png&refer=http%3A%2F%2Fwww.pianshen.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1655542961&t=63d9a67cfb37bf72b0907125e124281a",
      url2:"https://img0.baidu.com/it/u=1001309324,439806935&fm=253&fmt=auto&app=138&f=JPEG?w=667&h=500"
    }
  }
}
</script>

<style scoped>
.cardWord{
  font-family: XYSJW;

  animation: zoomInDown; /* referring directly to the animation's @keyframe declaration */
  animation-duration: 1s; /* don't forget to set a duration! */
}
</style>