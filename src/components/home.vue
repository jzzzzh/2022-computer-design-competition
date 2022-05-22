<template>
  <view class="homeBackground">
    <div class="bgBlurred">
      <h1  class="animate__animated animate__flipInX" id="HeadLine1" style="margin-top: 40px; font-family: XYSJW">智投</h1>
      <h2  class="animate__animated animate__flipInX" id="HedaLine2" style="margin-top: 40px; font-family: XYSJW">基于LSTM-GA的智能量化投资系统</h2>
      <el-button type="primary" style="margin-top: 40px;" size="large" @click="goToPredict">快速开始<el-icon class="el-icon--right" :size="20"> <Mouse/></el-icon> </el-button>
    </div>
  </view>
</template>

<script>
import 'animate.css';
import {Mouse, View} from '@element-plus/icons-vue'
export default {
  /* eslint-disable */
  name: "home",
  components: {View, Mouse},
  methods: {
    goToPredict() {
      this.$router.push("/Predict");
    },
  }
}

</script>

<style scoped>
#HeadLine1{
  /*position: fixed;*/
  /*top: 200px;*/
  /*font-family: 'PingFang SC';*/
  font-size: 60px;
  color: beige;
  display: flex;
  align-content: center;
  justify-content: center;

}
.bgBlurred{
  width: 100%;
  height: 100%;
  position: absolute;
  background: rgba(0, 0, 0, 0.5);
}
#HedaLine2{
  /*font-family: YHJW;*/
  color: beige;
}

.homeBackground{
  background: url("../assets/background.jpg");
  /*background-size: 100px 100px;*/
  width:100%;
  height:100%;
  position:fixed;
  top: 80px;
  right: 0;
  background-size:100% 100%;
}

</style>