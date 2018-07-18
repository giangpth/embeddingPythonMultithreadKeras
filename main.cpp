#include <Python.h>
#include <thread>
#include "test.h"
#include <pthread.h>
#include <QDebug>

using namespace std;
//function to assign a new thread state for combination of new sub interpreter and thread then acquire gil lock and run code. Release gil lock when done.

void* do_stuff_in_thread(PyInterpreterState* interp/*, PyObject* ins, int tid*/) //this function is not work well when call in thread
{
    // acquire the GIL
    PyEval_AcquireLock();
    qDebug() << "Acquire lock";

    // create a new thread state for the the sub interpreter interp
    PyThreadState* ts = PyThreadState_New(interp);

    // make ts the current thread state
    PyThreadState_Swap(ts);
    qDebug() << "Swap from current state to this state";
    // at this point:
    // 1. You have the GIL
    // 2. You have the right thread state - a new thread state (this thread was not created by python) in the context of interp

    // PYTHON WORK HERE
//    qDebug() << "Thread " << tid;
    PyInit_test();
    PyObject* vgg = createVGG();
    if(PyErr_Occurred())
    {
        PyErr_Print();
    }
    callPredict(vgg);
    if(PyErr_Occurred())
    {
        PyErr_Print();
    }
    qDebug() << "Run python code done";

    // release ts
    PyThreadState_Swap(NULL);

    // clear and delete ts
    qDebug() << "Clear and delete state";
    PyThreadState_Clear(ts);
    PyThreadState_Delete(ts);

    // release the GIL
    qDebug() << "Release lock";
    PyEval_ReleaseLock();
}

void* myfunc(PyThreadState* maints, PyObject* vgg, int id)
{
    PyInterpreterState* mainInter = maints->interp;
    PyThreadState* newth = PyThreadState_New(mainInter);

    PyEval_AcquireLock(); //acquire gil
    PyThreadState_Swap(newth);
    callPredict(vgg);
    if(PyErr_Occurred())
    {
        PyErr_Print();
    }
    PyRun_SimpleString("print('print in new thread')");
    qDebug() << "Thread id " << id;
    PyThreadState_Swap(maints);
    PyThreadState_Clear(newth);
    PyThreadState_Delete(newth);

}

void* myfunc2(PyInterpreterState* mainInter, PyObject* vgg, int id) //another function is not work well when call in thread
{
    PyThreadState* newth = PyThreadState_New(mainInter);

    PyEval_AcquireLock(); //acquire gil
    PyThreadState_Swap(newth);
    callPredict(vgg);
    if(PyErr_Occurred())
    {
        PyErr_Print();
    }
    PyRun_SimpleString("print('print in new thread')");
    qDebug() << "Thread id " << id;
    PyThreadState_Swap(NULL);
    PyThreadState_Clear(newth);
    PyThreadState_Delete(newth);

}

int main(int argc, char** argv)
{
    Py_Initialize(); // Initialize Python main interpreter
    PyEval_InitThreads(); //Init and acquire GIL, should call in main thread to create and acquire GIL for main thread

    PyInit_test();
    PyObject* vgg = createVGG();
    if(PyErr_Occurred())
    {
        PyErr_Print();
    }
    time_t t1 = clock();

    PyThreadState* maints = PyThreadState_Get(); //get thread state of main thread


    PyEval_ReleaseLock(); //main thread release gil
    qDebug() << "Release lock of main thread done";

    thread th1(myfunc, maints, vgg, 1);
    th1.join();


    callPredict(vgg);

    if(PyErr_Occurred())
    {
        PyErr_Print();
    }


    time_t t2 = clock();
    qDebug() << t2 -t1;

    Py_Finalize();

}
