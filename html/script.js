document.write("haha");
var vib = 24;
vib = vib + 1;
var he = "这是一段文字";
document.write("<br/>");
var aa = Math.pow(vib,2);
document.write(aa);
//document.write(he.length);
document.write("<br/>");
var bb = Math.sqrt(4,2);
document.write(bb);
document.write("<br/>");
document.write(Math.random());
document.write("<br/>");
document.write(Math.round(Math.random() * 10));
//弹出对话框要求输入
//var password = prompt("请输入登录密码");
document.write("<hr/>");
//document.write("您的登录密码是" + password +"对吧");
document.write("<h1>一台简单的计算器</h1>");
//转换成整数
document.write(parseInt(5.33));
//转换成浮点数
//parseFloat()

//函数
function hello(name){
    document.write("你好"+ name);
}
hello("老白");
document.write("<hr/>");
//获取input中的值
function value_input(){
    var value1 = document.getElementById('value').value;
    alert("您的邮箱是"+value1);
}
