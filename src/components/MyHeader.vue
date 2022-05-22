<template>
  <img class="icon" @click="goToHome" src="../assets/icon.png" style="position: absolute">
  <h1 class="text">智投——基于LSTM-GA的量化投资系统</h1>
  <el-row class="headRow" style="position: absolute">
    <el-button type="primary" @click="goToHome" plain>首页</el-button>
    <el-button type="primary" @click="goToTest" plain>介绍</el-button>
    <el-button type="primary" plain @click="goToPredict">投资建议</el-button>
    <el-button type="primary" plain @click="goToAPItro">API</el-button>
    <el-button type="primary" plain @click="download">下载</el-button>
</el-row>
</template>

<script>
export default {
  name: "MyHeader",
  data() {
    return {
      passage: "",
      abstract: ""
    }},
  methods: {
    goToHome(){
      this.$router.push("/home");
    },
    goToTest(){
      this.$router.push("/intro");
    },
    goToPredict(){
      this.$router.push("/predict");
    },
    goToAPItro(){
      this.$router.push("/APIntro");
    },
    download(){
      console.log("download");
      location = 'http://127.0.0.1:5000/download';
    }
  }
}
</script>

<style scoped>
.headRow{
  position: fixed;
  top: 30px;
  right: 10px;
  margin-right: -10px;
}
.icon{
  width: 60px;
  height: 60px;
  position: fixed;
  left: 70px;
  top: 10px;
}
.text{
  font-family: JYDJW;
  font-size: 30px;
}
</style>