<template>
  <el-tabs
      v-model="activeName"
      type="card"
      class="demo-tabs"
  >
    <el-tab-pane label="POST请求使用方法" name="first">
      <h1>对POST操作</h1>
      <p>POST获取相应请求介绍（以vue3+axios为例）</p>

      <el-space :fill="false" wrap >
        <el-card class="box-card" style="width: 700px">
          <template #header>
            <div class="card-header">
              <span>post网址</span>
            </div>
          </template>
          <p>www.lonelydone.com/LSTM-GA/predict</p>
        </el-card>
      </el-space>
      <br>
      <el-space :fill="false" style="margin-top: 30px; " wrap>
        <el-card class="box-card" style="width: 700px">
          <template #header>
            <div class="card-header">
              <span>传输数据</span>
            </div>
          </template>
          <p>
            item1: 商品一的名称（同选择名）,
            <br>
            item2: 商品二的名称（同选择名）,
            <br>
            USD: 现金金额,
            <br>
            item1money:商品一的持有量,
            <br>
            item2money:商品二的持有量,
            <br>
            MoneyForm:单位(0 美元， 1 人民币）,
            <br>
            choose:投资政策选择（0 稳健， 1 经典， 2 激进）
          </p>
        </el-card>
      </el-space>
      <br>
      <el-space :fill="false" style="margin-top: 30px; " wrap>
        <el-card class="box-card" style="width: 700px">
          <template #header>
            <div class="card-header">
              <span>返回类型</span>
            </div>
          </template>
          <p>
            whatToDo：需要做什么
            <br>
            buyItem1：买卖商品1的比例（0-1）
            <br>
            buyItem2：买卖商品2的比例（0-1）
          </p>
        </el-card>
      </el-space>
    </el-tab-pane>
<!--    <el-tab-pane label="GET请求使用方法" name="second">Config</el-tab-pane>-->
    <el-tab-pane label="PYTHON本地使用" name="third">
      <h1>PYTHON本地使用</h1>
      <p>在GitHub中下载源码</p>
      <a style="text-decoration:none;" href="https://github.com/jzzzzh/2022-American-Mathematical-Modeling-Competition/blob/main/LSTM_GA_Predict.py" target="_blank">
        <el-button type="primary" >点击获取源码</el-button>
      </a>

      <el-divider />
     <h1>函数介绍</h1>
      <el-space :fill="false" style="margin-top: 30px; " wrap>
        <el-card class="box-card" style="width: 700px">
          <template #header>
            <div class="card-header">
              <span>start_pred   开始预测</span>
            </div>
          </template>
          <p>

            mymoney：我手上的钱[现金，项目1，项目2]
            <br>
            testNum：预测次数
            <br>
            firstCommission：项目1手续费
            <br>
            secondCommission：项目2手续费
          </p>
        </el-card>
      </el-space>
      <el-space :fill="false" style="margin-top: 30px; " wrap>
        <el-card class="box-card" style="width: 700px">
          <template #header>
            <div class="card-header">
              <span>start_EC  开始进化计算</span>
            </div>
          </template>
          <p>
            testNum：进化次数
            <br>
            firstCommission：项目1的手续费
            <br>
            secondCommission：项目2手续费
            <br>
            ECtestNum：进化个体数
            <br>
            ECtesttimes：进化次数
            <br>
            electRate：精英主义概率
            <br>
            MutationRate：变异概率
            <br>
            firstmoney:我手上的钱[现金，项目1，项目2]
          </p>
        </el-card>
      </el-space>
      <el-space :fill="false" style="margin-top: 30px; " wrap>
        <el-card class="box-card" style="width: 700px">
          <template #header>
            <div class="card-header">
              <span>get_ffinal_pred  最终预测</span>
            </div>
          </template>
          <p>
            mymoney：我手上的钱[现金，项目1，项目2]
            <br>
            testNum：预测次数
            <br>
            firstCommission：项目1手续费
            <br>
            secondCommission：项目2手续费
            <br>
            numListfirst：项目1的分析数据
            <br>
            numListsecond：项目2的分析数据
          </p>
        </el-card>
      </el-space>

    </el-tab-pane>

    <el-tab-pane label="软件使用说明" name="fourth">


      <h1>软件使用说明</h1>
      <el-space :fill="false" style="margin-top: 30px; " wrap>
        <el-card class="box-card" style="width: 700px">
          <template #header>
            <div class="card-header">
              <span>软件使用说明</span>
            </div>
          </template>
          <a style="text-decoration:none;" href="https://cloud.06dn.com/s/9ARvcv" target="_blank">
            <el-button type="primary" >点击查看教学视频</el-button>
          </a>
        </el-card>
      </el-space>
    </el-tab-pane>

  </el-tabs>
</template>

<script>
export default {
  name: "APIntro",
  data(){
    return{
      activeName:"first"
    }
  }
}
</script>

<style scoped>
.demo-tabs > .el-tabs__content {
  padding: 32px;
  color: #6b778c;
  font-size: 32px;
  font-weight: 600;
}
</style>