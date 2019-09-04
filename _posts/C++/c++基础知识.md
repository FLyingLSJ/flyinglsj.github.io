[TOC]

### 基础知识

- 对象 -** 对象具有状态和行为。例如：一只狗的状态 - 颜色、名称、品种，行为 - 摇动、叫唤、吃。对象是类的实例。
- **类 -** 类可以定义为描述对象行为/状态的模板/蓝图。
- **方法 -** 从基本上说，一个方法表示一种行为。一个类可以包含多个方法。可以在方法中写入逻辑、操作数据以及执行所有的动作。
- **即时变量 -** 每个对象都有其独特的即时变量。对象的状态是由这些即时变量的值创建的。

```c++
#include <iostream> // 头文件
using namespace std; // 告诉编译器使用 std 命名空间。命名空间是 C++ 中一个相对新的概念。
 
// main() 是程序开始执行的地方
 
int main() //程序从这里开始执行。
{
   cout << "Hello World"; // 输出 Hello World
   return 0;
}
```

- 以分号为结束符
- 使用花括号 {}
- 变量定义：一个标识符以字母 A-Z 或 a-z 或下划线 _ 开始，后跟零个或多个字母、下划线和数字（0-9）。大小写敏感
- 关键字
- 三字符组

| 三字符组 | 替换 |
| :------- | :--- |
| ??=      | #    |
| ??/      | \    |
| ??'      | ^    |
| ??(      | [    |
| ??)      | ]    |
| ??!      | \|   |
| ??<      | {    |
| ??>      | }    |
| ??-      | ~    |

- 空格符：

### 注释

```c++
// 注释

/* 这是注释 */
 
/* C++ 注释也可以
 * 跨行
 */
```

### 数据类型

| 类型     | 关键字  |
| :------- | :------ |
| 布尔型   | bool    |
| 字符型   | char    |
| 整型     | int     |
| 浮点型   | float   |
| 双浮点型 | double  |
| 无类型   | void    |
| 宽字符型 | wchar_t |

不同数据类型占据的内存大小（视系统而定）

| 类型               | 位            | 范围                                                    |
| :----------------- | :------------ | :------------------------------------------------------ |
| char               | 1 个字节      | -128 到 127 或者 0 到 255                               |
| unsigned char      | 1 个字节      | 0 到 255                                                |
| signed char        | 1 个字节      | -128 到 127                                             |
| int                | 4 个字节      | -2147483648 到 2147483647                               |
| unsigned int       | 4 个字节      | 0 到 4294967295                                         |
| signed int         | 4 个字节      | -2147483648 到 2147483647                               |
| short int          | 2 个字节      | -32768 到 32767                                         |
| unsigned short int | 2 个字节      | 0 到 65,535                                             |
| signed short int   | 2 个字节      | -32768 到 32767                                         |
| long int           | 8 个字节      | -9,223,372,036,854,775,808 到 9,223,372,036,854,775,807 |
| signed long int    | 8 个字节      | -9,223,372,036,854,775,808 到 9,223,372,036,854,775,807 |
| unsigned long int  | 8 个字节      | 0 to 18,446,744,073,709,551,615                         |
| float              | 4 个字节      | +/- 3.4e +/- 38 (~7 个数字)                             |
| double             | 8 个字节      | +/- 1.7e +/- 308 (~15 个数字)                           |
| long double        | 16 个字节     | +/- 1.7e +/- 308 (~15 个数字)                           |
| wchar_t            | 2 或 4 个字节 | 1 个宽字符                                              |

- typedef 声明：

```c++
// typedef 为一个已有的类型取一个新的名字。
// 使用格式：typedef type newname; 
typedef int feet; // feet 是 int 的另一个名称
feet dis // 这是合法的，因为这里的 feet 相当于 int
```

- 枚举类型

```c++
// 使用格式如下：
/*
enum 枚举名{ 
     标识符[=整型常数], 
     标识符[=整型常数], 
... 
    标识符[=整型常数]
} 枚举变量;
*/
enum color { red, green, blue } c;
c = blue; // 定义了一个颜色枚举，变量 c 的类型为 color。最后，c 被赋值为 "blue"。

// 第一个名称的值为 0，第二个名称的值为 1，第三个名称的值为 2，以此类推。但是，您也可以给名称赋予一个特殊的值，只需要添加一个初始值即可。例如，在下面的枚举中，green 的值为 5。
enum color { red, green=5, blue }; // 这里的 blue 是 6， 而 red 依然是 0

```

### 变量类型

|  类型   |                             描述                             |
| :-----: | :----------------------------------------------------------: |
|  bool   |                    存储值 true 或 false。                    |
|  char   |          通常是一个字符（八位）。这是一个整数类型。          |
|   int   |               对机器而言，整数的最自然的大小。               |
|  float  | 单精度浮点值。单精度是这样的格式，1位符号，8位指数，23位小数。![img](https://www.runoob.com/wp-content/uploads/2014/09/v2-749cc641eb4d5dafd085e8c23f8826aa_hd.png) |
| double  | 双精度浮点值。双精度是1位符号，11位指数，52位小数。![img](https://www.runoob.com/wp-content/uploads/2014/09/v2-48240f0e1e0dd33ec89100cbe2d30707_hd.png) |
|  void   |                       表示类型的缺失。                       |
| wchar_t |                         宽字符类型。                         |

```c++
// 变量声明
int    i, j, k;
char   c, ch;
float  f, salary;
double d;

// 变量声明并初始化 
extern int d = 3, f = 5;    // d 和 f 的声明 
int d = 3, f = 5;           // 定义并初始化 d 和 f
byte z = 22;                // 定义并初始化 z
char x = 'x';               // 变量 x 的值为 'x'
```

```c++
using namespace std;
// 变量声明
extern int a, b;
extern int c;
extern float f;
int main ()
{
    // 变量定义
    int a, b;
    int c;
    float f;
    // 实际初始化
    a = 10;
    b = 20;
    c = a + b;
    cout << c << endl ;
    f = 70.0/3.0;
    cout << f << endl ;
    return 0;
}

// 结果：
// 30
// 23.3333
```

- 左值，右值
  - **左值（lvalue）：**指向内存位置的表达式被称为左值（lvalue）表达式。左值可以出现在赋值号的左边或右边。

  - **右值（rvalue）：**术语右值（rvalue）指的是存储在内存中某些地址的数值。右值是不能对其进行赋值的表达式，也就是说，右值可以出现在赋值号的右边，但不能出现在赋值号的左边。

  - ```c++
    int a = 10; // 正确的
    10 = 20 // 错误的
    ```

### 变量作用域

- 在函数或一个代码块内部声明的变量，称为局部变量。
- 在函数参数的定义中声明的变量，称为形式参数。
- 在所有函数外部声明的变量，称为全局变量。

```c++
// 全局变量与局部变量
#include <iostream>
using namespace std;
// 全局变量声明
int g;
int main ()
{
  // 局部变量声明
  int a, b;
  // 实际初始化
  a = 10;
  b = 20;
  g = a + b;
 
  cout << g;
 
  return 0;
}
```

在程序中，局部变量和全局变量的名称可以相同，但是在函数内，局部变量的值会覆盖全局变量的值。

```c++
#include <iostream>
using namespace std;
 
// 全局变量声明
int g = 20;
 
int main ()
{
  // 局部变量声明
  int g = 10;
 
  cout << g;
 
  return 0;
}
// 输出结果： 10 // 因为局部变量会将全局变量给覆盖掉
```

局部变量需要自己初始化，但是全局变量会被系统自动初始化为以下的值，**正确地初始化变量是一个良好的编程习惯，否则有时候程序可能会产生意想不到的结果。**

| 数据类型 | 初始化默认值 |
| :------- | :----------- |
| int      | 0            |
| char     | '\0'         |
| float    | 0            |
| double   | 0            |
| pointer  | NULL         |

### 常量

常量是固定值，在程序执行期间不会改变。这些固定的值，又叫做**字面量**。

常量可以是任何的基本数据类型，可分为整型数字、浮点数字、字符(单引号)、字符串(双引号)和布尔值。

常量就像是常规的变量，只不过常量的值在定义后不能进行修改。

常量定义：

- #define 预处理器 `#define identifier value`
- const 关键字 `const type variable = value;`

把常量定义为大写字母形式，是一个很好的编程实践。

### 运算符

| 算术运算符 |      |
| :--------: | ---- |
| 关系运算符 |      |
| 逻辑运算符 |      |
|  位运算符  |      |
| 赋值运算符 |      |
| 杂项运算符 |      |

### 循环

```c++
// while
while(condition)
{
   statement(s);
}

// for
for ( init; condition; increment )
{
   statement(s);
}

// do ... while 
do
{
   statement(s);  // 至少会被执行一次

}while( condition );

```

### 判断

```c++
// if
if(boolean_expression){
   // 如果布尔表达式为真将执行的语句
}

// if... else
if(boolean_expression){
   // 如果布尔表达式为真将执行的语句
}
else{
   // 如果布尔表达式为假将执行的语句
}

// if .. else if .. else
if(boolean_expression 1)
{
   // 当布尔表达式 1 为真时执行
}
else if( boolean_expression 2){
   // 当布尔表达式 2 为真时执行
}
else if( boolean_expression 3){
   // 当布尔表达式 3 为真时执行
}
else {
   // 当上面条件都不为真时执行
}

// switch 可以嵌套
switch(expression){
    case constant-expression  :
       statement(s);
       break; // 可选的
    case constant-expression  :
       statement(s);
       break; // 可选的
 
    // 您可以有任意数量的 case 语句
    default : // 可选的
       statement(s);
}

// 表达式的值是由 Exp1 决定的。如果 Exp1 为真，则计算 Exp2 的值，结果即为整个 ? 表达式的值。如果 Exp1 为假，则计算 Exp3 的值，结果即为整个 ? 表达式的值。
Exp1 ? Exp2 : Exp3;
```

- **switch** 语句中的 **expression** 必须是一个整型或枚举类型，或者是一个 class 类型，其中 class 有一个单一的转换函数将其转换为整型或枚举类型。
- 在一个 switch 中可以有任意数量的 case 语句。每个 case 后跟一个要比较的值和一个冒号。
- case 的 **constant-expression** 必须与 switch 中的变量具有相同的数据类型，且必须是一个常量或字面量。
- 当被测试的变量等于 case 中的常量时，case 后跟的语句将被执行，直到遇到 **break** 语句为止。
- 当遇到 **break** 语句时，switch 终止，控制流将跳转到 switch 语句后的下一行。
- 不是每一个 case 都需要包含 **break**。如果 case 语句不包含 **break**，控制流将会 *继续* 后续的 case，直到遇到 break 为止。
- 一个 **switch** 语句可以有一个可选的 **default** case，出现在 switch 的结尾。default case 可用于在上面所有 case 都不为真时执行一个任务。default case 中的 **break** 语句不是必需的。

### 函数

```c++
// 函数声明
return_type function_name( parameter list );

int max(int num1, int num2);  // 例子
int max(int, int); // 同上，可以没有参数名

// 函数定义
return_type function_name( parameter list ){
   body of the function
}
/*
	返回类型：一个函数可以返回一个值。return_type 是函数返回的值的数据类型。有些函数执行所需的操作而不返回值，在这种情况下，return_type 是关键字 void。
	函数名称：这是函数的实际名称。函数名和参数列表一起构成了函数签名。
	参数：参数就像是占位符。当函数被调用时，您向参数传递一个值，这个值被称为实际参数。参数列表包括函数参数的类型、顺序、数量。参数是可选的，也就是说，函数可能不包含参数。
	函数主体：函数主体包含一组定义函数执行任务的语句。
*/
int max(int num1, int num2){
    body;
}

```

函数调用

- 传值调用：该方法把参数的实际值复制给函数的形式参数。在这种情况下，修改函数内的形式参数对实际参数没有影响。

```c++
// 函数定义
void swap(int x, int y){
   int temp;
   temp = x; /* 保存 x 的值 */
   x = y;    /* 把 y 赋值给 x */
   y = temp; /* 把 x 赋值给 y */
   return;
}
```

- 指针调用：该方法把参数的地址复制给形式参数。在函数内，该地址用于访问调用中要用到的实际参数。这意味着，修改形式参数会影响实际参数。

```c++
// 函数定义
void swap(int *x, int *y)  // 两个指针变量
{
   int temp;
   temp = *x;    /* 保存地址 x 的值 */
   *x = *y;        /* 把 y 赋值给 x */
   *y = temp;    /* 把 x 赋值给 y */
   return;
}
```

- 引用调用：该方法把参数的引用复制给形式参数。在函数内，该引用用于访问调用中要用到的实际参数。这意味着，修改形式参数会影响实际参数。

```c++
// 函数定义
void swap(int &x, int &y)
{
   int temp;
   temp = x; /* 保存地址 x 的值 */
   x = y;    /* 把 y 赋值给 x */
   y = temp; /* 把 x 赋值给 y  */
  
   return;
}
```

### 数组

```c++
// 一维数组
// type arrayName [ arraySize ]; 声明数组
// type：数据类型 arraySize：数组大小（>0）
double balance[10]; // 声明数组

double balance[5] = {1000.0, 2.0, 3.4, 7.0, 50.0}; // 初始化
double balance[] = {1000.0, 2.0, 3.4, 7.0, 50.0};  // 省略数组大小参数，故初始化的数据容量就是数组的大小


//多维数组初始化
int a[3][4] = {  
 {0, 1, 2, 3} ,   /*  初始化索引号为 0 的行 */
 {4, 5, 6, 7} ,   /*  初始化索引号为 1 的行 */
 {8, 9, 10, 11}   /*  初始化索引号为 2 的行 */
};

int a[3][4] = {0,1,2,3,4,5,6,7,8,9,10,11}; // 效果同上
```

数组指针

- 数组名本身是一个指针，数组第一个值的地址也是这个数组的地址

### 字符串

字符串操作的库 `#include <cstring>` 还有面向对象编程的 String 类 `#include <string>`

### 指针

```c++
//常见的操作：定义一个指针变量、把变量地址赋值给指针、访问指针变量中可用地址的值。
int    *ip;    /* 一个整型的指针 */
double *dp;    /* 一个 double 型的指针 */
float  *fp;    /* 一个浮点型的指针 */
char   *ch;    /* 一个字符型的指针 */


int  var = 20;   // 实际变量的声明
int  *ip;        // 指针变量的声明 
ip = &var;       // 在指针变量中存储 var 的地址
```

Null 指针：

- 在变量声明的时候，如果没有确切的地址可以赋值，为指针变量赋一个 NULL 值是一个良好的编程习惯。赋为 NULL 值的指针被称为**空**指针。`int *p = Null` 

指针的算术运算：四种算术运算：++、--、+、-

### 引用



### 时间与日期

`ctime` 

有四个与时间相关的类型：**clock_t、time_t、size_t** 和 **tm**。类型 clock_t、size_t 和 time_t 能够把系统时间和日期表示为某种整数。