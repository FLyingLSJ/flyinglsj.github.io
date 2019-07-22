ironpython 解释器 https://ironpython.net/documentation/

博客：https://blog.csdn.net/u010302494/article/details/77929364

https://www.python.org/downloads/

https://www.cnblogs.com/nickli/archive/2011/02/27/1966144.html

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IronPython.Hosting;
using Microsoft.Scripting.Hosting;
 
namespace demo
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                ScriptRuntime pyRunTime = Python.CreateRuntime();
                dynamic obj = pyRunTime.UseFile("hello.py");

                Console.WriteLine(obj.welcome("nick"));
                Console.ReadKey();
            
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}

```

https://code.msdn.microsoft.com/windowsdesktop/C-and-Python-interprocess-171378ee

```c#
using System; 
using System.IO; 
using System.Diagnostics; 
 
namespace CallPython 
{ 
    /// <summary> 
    /// Used to show simple C# and Python interprocess communication 
    /// Author      : Ozcan ILIKHAN 
    /// Created     : 02/26/2015 
    /// Last Update : 04/30/2015 
    /// </summary> 
    class Program 
    { 
        static void Main(string[] args) 
        { 
            // full path of python interpreter 
            string python = @"C:\Continuum\Anaconda\python.exe"; 
             
            // python app to call 
            string myPythonApp = "sum.py"; 
 
            // dummy parameters to send Python script 
            int x = 2; 
            int y = 5; 
 
            // Create new process start info 
            ProcessStartInfo myProcessStartInfo = new ProcessStartInfo(python); 
 
            // make sure we can read the output from stdout 
            myProcessStartInfo.UseShellExecute = false; 
            myProcessStartInfo.RedirectStandardOutput = true; 
 
            // start python app with 3 arguments  
            // 1st arguments is pointer to itself,  
            // 2nd and 3rd are actual arguments we want to send 
            myProcessStartInfo.Arguments = myPythonApp + " " + x + " " + y; 
 
            Process myProcess = new Process(); 
            // assign start information to the process 
            myProcess.StartInfo = myProcessStartInfo; 
 
            Console.WriteLine("Calling Python script with arguments {0} and {1}", x,y); 
            // start the process 
            myProcess.Start(); 
 
            // Read the standard output of the app we called.  
            // in order to avoid deadlock we will read output first 
            // and then wait for process terminate: 
            StreamReader myStreamReader = myProcess.StandardOutput; 
            string myString = myStreamReader.ReadLine(); 
             
            /*if you need to read multiple lines, you might use: 
                string myString = myStreamReader.ReadToEnd() */           
 
            // wait exit signal from the app we called and then close it. 
            myProcess.WaitForExit(); 
            myProcess.Close(); 
 
            // write the output we got from python app 
            Console.WriteLine("Value received from script: " + myString); 
             
        } 
    } 
} 
```

