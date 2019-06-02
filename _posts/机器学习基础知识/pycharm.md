### Pycharm 安装及激活

https://blog.csdn.net/qq_41915690/article/details/89184917

https://blog.csdn.net/qq_15698613/article/details/86502371

### 扩展工具

[代码规范](https://zhuanlan.zhihu.com/p/59763076)

1. 字体

`settings->Editor->Font`

- Font: Courier New 
- Fallback font: DFKai-SB

### 提示信息设置

在 **File---settings---File and Code Templates---Python script** 脚本里添加:

```
#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:${USER}
@file: ${NAME}.py
@time: ${YEAR}/${MONTH}/${DAY}
"""
```

常见字段：

```
${PROJECT_NAME} - the name of the current project.
${NAME} - the name of the new file which you specify in the New File dialog box during the file creation.
${USER} - the login name of the current user.
${DATE} - the current system date.
${TIME} - the current system time.
${YEAR} - the current year.
${MONTH} - the current month.
${DAY} - the current day of the month.
${HOUR} - the current hour.
${MINUTE} - the current minute.
${PRODUCT_NAME} - the name of the IDE in which the file will be created.
${MONTH_NAME_SHORT} - the first 3 letters of the month name. Example: Jan, Feb, etc.
${MONTH_NAME_FULL} - full name of a month. Example: January, February, etc.
```

