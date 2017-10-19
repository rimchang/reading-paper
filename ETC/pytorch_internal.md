### A Tour of PyTorch Internals (Part I)
The fundamental unit in PyTorch is the Tensor. This post will serve as an overview for how we implement Tensors in PyTorch, such that the user can interact with it from the Python shell. In particular, we want to answer four main questions

Pytoch의 가장 근본적인 unit은 Tensor입니다. 이 포스트는 Pytorch의 Tensor가 어떻게 구현되어져 있고 어떻게 Python shell과 상호작용 하는지에 대한 것을 설명합니다. 특히 밑의 4가지 답변에 대해 답할 수 있도록 할 것입니다.

 1. How does PyTorch extend the Python interpreter to define a Tensor type that can be manipulated from Python code?
 2. How does PyTorch wrap the C libraries that actually define the Tensor's properties and methods?
 3. How does PyTorch cwrap work to generate code for Tensor methods?
 4. How does PyTorch's build system take all of these components to compile and generate a workable application?

1. Pytorch는 Python interpreter를 어떻게 확장하여 Tensor type을 정의하는가?
2. Tensor's properties, method가 정의되어 있는 C 라이브러리를 어떻게 wrap 하는가?
3. Tensor methods에 대한 코드를 생성하게 위해 어떻게 cwrap이 작동하는가
4. 이러한 component를 받아서 어떻게 컴파일 하고 application을 작동하도록 하는가?


#### Extending the Python Interpreter
PyTorch defines a new package `torch`. In this post we will consider the `._C` module. This module is known as an "extension module" - a Python module written in C. Such modules allow us to define new built-in object types (e.g. the `Tensor`) and to call C/C++ functions.

Pytorch는 새로운 패키지 `torch`를 정의합니다. 이 포스트에서는 `.C` 에 대해서 다룰 것입니다. 이 모듈은 일종의 C로 짜여진 Python module로 일종의 "extension module" 로 알려져 있습니다. 이러한 모듈은 새로운 built-in object type `tensor`를 사용할 수 있게하고 C/C++ 함수를 호출 할수 있도록 합니다.

The `._C` module is defined in `torch/csrc/Module.cpp`. The `init_C()` / `PyInit__C()` function creates the module and adds the method definitions as appropriate. This module is passed around to a number of different `__init()` functions that add further objects to the module, register new types, etc.

`._C` 모듈은 `torch/csrc/Module.cpp`에 정의되어져 있습니다. `init_C()` / `PyInit__C()` 함수가 모듈을 생성하고 적절한 방법으로 method 정의를 추가합니다. 이 모듈은 `__init()` 함수에 의해 통과되어 새로운 object를 module에 추가합니다.

One collection of these `__init()` calls is the following:

`__init()` 이 불리는 예는 다음과 같습니다.

```
ASSERT_TRUE(THPDoubleTensor_init(module));
ASSERT_TRUE(THPFloatTensor_init(module));
ASSERT_TRUE(THPHalfTensor_init(module));
ASSERT_TRUE(THPLongTensor_init(module));
ASSERT_TRUE(THPIntTensor_init(module));
ASSERT_TRUE(THPShortTensor_init(module));
ASSERT_TRUE(THPCharTensor_init(module));
ASSERT_TRUE(THPByteTensor_init(module));
```
These `__init()` functions add the Tensor object for each type to the `._C` module so that they can be used in the module. Let's learn how these methods work.

이러한 `__init()` 함수는 각 tye의 `._C` 모듈을 Tensor object에 추가합니다. 이를 통해 이들은 module 안에서 사용될 수 있습니다. 이게 이러한 method가 어떻게 작동하는지 배워봅시다.

#### The THPTensor Type
Much like the underlying `TH` and `THC` libraries, PyTorch defines a "generic" Tensor which is then specialized to a number of different types. Before considering how this specialization works, let's first consider how defining a new type in Python works, and how we create the generic `THPTensor` type.

`TH`, `THC` 의 이름이 의미하듯이 Pytorch는 "generic" Tensor를 정의합니다. 그다음 이는 몇몇개의 다른 type들로 specialized 됩니다. 이러한 specialization이 어떻게 작동하는지 고려하기 전에 먼저 Python 에서의 새로운 type의 정의를 어떻게 하는지 알아보고 generic `THPTensor` type을 어떻게 만드는지 알아봅시다.


The Python runtime sees all Python objects as variables of type `PyObject *`, which serves as a "base type" for all Python objects. Every Python type contains the refcount for the object, and a pointer to the object's *type object*. The type object determines the properties of the type. For example, it might contain a list of methods associated with the type, and which C functions get called to implement those methods. The object also contains any fields necessary to represent its state.

Python 실행시간에서 모든 Python object는 일종의 `PyObject *` type (PyObject의 포인터 타입) 입니다. 이는 모든 Python objects의 "base type"을 제공합니다. 모든 Python type은 object에 대한 refcount를 포함하며 object의 *type object*의 포인터를 포함합니다. object의 타입은 그 type의 properties를 결정합니다. 예를들어 특정 type과 관련된 method의 리스르르 포함할 수 있으며 그에 해당하는 C 로 구현된 함수가 불려집니다. 이 object는 그의 state를 나타낼 수 있는 필요한 fields를 포함합니다.

The formula for defining a new type is as follows:

새로운 type을 정의하기 위한 방법은 다음과 같습니다.

 1. Create a struct that defines what the new object will contain
 2. Define the type object for the type
 
 1. 새로운 object가 포함해야할 것을 정의한 struct를 만듭니다
 2. type을 위한 type object를 정의합니다.

The struct itself could be very simple. Inn Python, all floating point types are actually objects on the heap. The Python float struct is defined as:

이 struct는 그자체로 매우 간단합니다. Python에서 모든 floating point type은 heap에서의 object입니다. Python float struct는 밑처럼 정의됩니다.


```
typedef struct {
    PyObject_HEAD
    double ob_fval;
} PyFloatObject;
```
The `PyObject_HEAD` is a macro that brings in the code that implements an object's reference counting, and a pointer to the corresponding type object. So in this case, to implement a float, the only other "state" needed is the floating point value itself.

`PyObject_HEAD`은 macro로써 object의 reference counting, 해당 type object에 대한 포인터를 가져옵니다. 이 경우에 float을 구현하기 위해서 floating point value 만을 나타내는 state만을 필요로 합니다.

Now, let's see the struct for our `THPTensor` type:

이제 `THPTensor` type의 struct를 정의합시다.

```
struct THPTensor {
    PyObject_HEAD
    THTensor *cdata;
};
```
Pretty simple, right? We are just wrapping the underlying `TH` tensor by storing a pointer to it.

매우 간단하지 않습니까? 우리는 `TH` tensor를 저장하는 pointer를 그냥 wrapping 했습니다.

The key part is defining the "type object" for a new type. An example definition of a type object for our Python float takes the form:

이것이 새로운 tpye에 대한 "type object"를 정의하는 주요 부분입니다. 하나의 예로 우리의 Python float을 정의하는 type object는 다음과 같습니다.

```
static PyTypeObject py_FloatType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "py.FloatObject",          /* tp_name */
    sizeof(PyFloatObject),     /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_as_async */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "A floating point number", /* tp_doc */
};
```
The easiest way to think of a *type object* is as a set of fields which define the properties of the object. For example, the `tp_basicsize` field is set to `sizeof(PyFloatObject)`. This is so that Python knows how much memory to allocate when calling `PyObject_New()` for a `PyFloatObject.` The full list of fields you can set is defined in `object.h` in the CPython backend:
https://github.com/python/cpython/blob/master/Include/object.h. 

*type object* 를 가장쉽게 생각하는 것은 object에 대한 properties를 정의한 field의 집합입니다. 예를들어  `tp_basicsize` field는 `sizeof(PyFloatObject)`의 설정입니다. 이것이 Python이 `PyFloatObject.`을 위해 `PyObject_New()` 을 호출할때 얼마나 메모리를 할당해야 할지를 알려줍니다. 설정할 수 있는 모든 리스트는 `object.h` 에 정의되어져 있습니다.


The type object for our `THPTensor` is `THPTensorType`, defined in `csrc/generic/Tensor.cpp`. This object defines the name, size, mapping methods, etc. for a `THPTensor`.

`THPTensor` 을 위한 type object는 `THPTensorType` 입니다. 이는 `csrc/generic/Tensor.cpp`에 정의되어져 있습니다. 이 object는 `THPTensor`을 위한 name, size, mapping methods을 정의합니다.


As an example, let's take a look at the `tp_new` function we set in the `PyTypeObject`:

하나의 예로 `tp_new` 함수를 살펴봅시다. 

```
PyTypeObject THPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  ...
  THPTensor_(pynew), /* tp_new */
};
```
The `tp_new` function enables object creation. It is responsible for creating (as opposed to initializing) objects of that type and is equivalent to the `__new()__` method at the Python level. The C implementation is a static method that is passed the type being instantiated and any arguments, and returns a newly created object.


`tp_new` 함수는 object의 생성을 가능하게 합니다. 이 함수는 object의 그 type에 대한 생성자의 역할을 맡고 있습니다. 이는 Python level에서의`__new()__` 와 동일합니다. C 구현은 static method로써 새로운 object를 반환합니다.

```
static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;

  THPTensorPtr self = (THPTensor *)type->tp_alloc(type, 0);
// more code below
```
The first thing our new function does is allocate the `THPTensor`. It then runs through a series of initializations based off of the args passed to the function. For example, when creating a `THPTensor` *x* from another `THPTensor` *y*, we set the newly created `THPTensor`'s `cdata` field to be the result of calling `THTensor_(newWithTensor)` with the *y*'s underlying `TH` Tensor as an argument. Similar constructors exist for sizes, storages, NumPy arrays, and sequences.

새로운 함수의 첫번째는 `THPTensor`를 할당하는 것입니다. 그다음 args를 통과시켜 일련의 생성자를 실행합니다. 예를들어 `THPTensor` type인 *x*를 다른 `THPTensor` type인 *y*를 생성할때 우리는 새롭게 생성되는 `THPTensor` 의 `cdata` field를 `THTensor_(newWithTensor)`의 호출 결과로 채웁니다. 이때 *y*는 underlying `TH` Tensor로 함수 인자로 들어갑니다. 비슷하게 sizes, storages, Numpy arrays, sequece에 대한 생성자가 존재합니다.

** Note that we solely use `tp_new`, and not a combination of `tp_new` and `tp_init` (which corresponds to the `__init()__` function).

** 우리가 `tp_new`를 거의 사용하지 않은 것을 주목하세요 그리고 `tp_new`과 `tp_init` 의 조합이 

The other important thing defined in Tensor.cpp is how indexing works. PyTorch Tensors support Python's **Mapping Protocol**. This allows us to do things like:

다른 Tensor.cpp에서 정의된 것중 중요한 것은 indexing이 어떻게 일어나는지에 대한 것입니다. Pytorch Tensors를 Python의 **Mapping Protocol**을 지원합니다. 이는 밑과 같은 일을 허락합니다.

```
x = torch.Tensor(10).fill_(1)
y = x[3] // y == 1
x[4] = 2
// etc.
```
** Note that this indexing extends to Tensor with more than one dimension

** Tensor에 대한 indeing이 다차원인 것에 

We are able to use the `[]`-style notation by defining the three mapping methods described here:
https://docs.python.org/3.7/c-api/typeobj.html#c.PyMappingMethods

우리는 `[]` 스타일에 대한 3가지의 mapping method에 대한 정의를 사용할 수 있습니다.


The most important methods are `THPTensor_(getValue)` and `THPTensor_(setValue)` which describe how to index a Tensor, for returning a new Tensor/Scalar, or updating the values of an existing Tensor in place. Read through these implementations to better understand how PyTorch supports basic tensor indexing.

또 다른 중요한 Method는 `THPTensor_(getValue)` and `THPTensor_(setValue)`입니다. 이는 Tensor를 어떻게 indexing 하는지, 새로운 Tensor/Scalar를 반환하고 이미 존재하는 Tensor의 값을 어떻게 업데이트 하는지를 설명합니다. 이들 구현을 읽음으로써 Pytorch가 기본 tensor indexing을 어떻게 하는지에 대한 이해를 하세요

#### Generic Builds (Part One)
We could spend a ton of time exploring various aspects of the `THPTensor` and how it relates to defining a new Python object. But we still need to see how the `THPTensor_(init)()` function is translated to the `THPIntTensor_init()` we used in our module initialization. How do we take our `Tensor.cpp` file that defines a "generic" Tensor and use it to generate Python objects for all the permutations of types? To put it another way, `Tensor.cpp` is littered with lines of code like:

우리는 `THPTensor`를 살펴보고 이가 어떻게 새로운 Python object를 정의하는것과 연관되어 있는지를 살펴보는데 많은 시간을 들였습니다. 하지만 우리는 `THPTensor_(init)()`가 어떻게 `THPIntTensor_init()`로 변환되는지 알아야 합니다. `Tensor.cpp`가 어떻게 "generic " Tensor를 정의하고 이를 이용해서 Python object를 생성할까요? 

```
return THPTensor_(New)(THTensor_(new)(LIBRARY_STATE_NOARGS));
```
This illustrates both cases we need to make type-specific:

* Our output code will call `THP<Type>Tensor_New(...)` in place of `THPTensor_(New)`
* Our output code will call `TH<Type>Tensor_new(...)` in place of `THTensor_(new)`

* 우리의 코드는 `THPTensor_(New)`대신에 `THP<Type>Tensor_New(...)` 호출 할 것입니다.
* 우리의 코드는 `THTensor_(new)` 대신에 `TH<Type>Tensor_new(...)`  호출 할 것입니다.

In other words, for all supported Tensor types, we need to "generate" source code that has done the above substitutions. This is part of the "build" process for PyTorch. PyTorch relies on Setuptools (https://setuptools.readthedocs.io/en/latest/) for building the package, and we define a `setup.py` file in the top-level directory to customize the build process. 

다른 말로 하면 모든 지원되는 Tensor type에 대해 우리는 "generate" source code가 필요합니다. 이는 위와 같은 것을 수행합니다. 이가 Pytorch의 "build" 과정중의 일부입니다. Pytorch는 패키지를 building 하는데 Setuptools 에 의존합니다. 우리는 `setup.py`에 build precess를 정의합니다.

One component building an Extension module using Setuptools is to list the source files involved in the compilation. However, our `csrc/generic/Tensor.cpp` file is not listed! So how does the code in this file end up being a part of the end product?

Setuptools을 이용한 Extenstion moudle의 하나의 요소는 컴파일된 source file의 리스트입니다. 그러나 우리의 `csrc/generic/Tensor.cpp` 는 리스트가 아닙니다. 그러면 이러한 것들이 어떻게 동작할까요?

Recall that we are calling the `THPTensor*` functions (such as `init`) from the directory above `generic`. If we take a look in this directory, there is another file `Tensor.cpp` defined. The last line of this file is important:

우리가 `THPTensor*` 함수를 호출한다는것을 기억해보세요 이는 `generic` 폴더의 상위에 있습니다. 만약 우리가 이 폴더를 본다면 여기에는 다른 `Tensor.cpp` 파일이 있습니다. 이 파일의 마지막 라인은 매우 중요합니다.

```
//generic_include TH torch/csrc/generic/Tensor.cpp
```
Note that this `Tensor.cpp` file is included in `setup.py`, but it is wrapped in a call to a Python helper function called `split_types`. This function takes as input a file, and looks for the "//generic_include" string in the file contents. If it is found, it generates a new output file for each Tensor type, with the following changes:

`setup.py` 안에 `Tensor.cpp`이 있다는 것에 주목합시다. 하지만 `split_types`이라는 Python helper 함수의 호출에 의해 wrap 됩니다. 이 함수는 file을 인풋으로 받아서 "//generic_include"의 내용을 살펴봅니다. 만약 찾았다면 이것은 새로운 각 Tensor type에 대한 새로운 output 파일을 생성합니다. 이때 다음과 같은 변화를 거칩니다.

 1. The output file is renamed to `Tensor<Type>.cpp`  결과 파일은 `Tensor<Type>.cpp` 로 재명명 됩니다.
 2. The output file is slightly modified as follows:
```
// Before:
//generic_include TH torch/csrc/generic/Tensor.cpp

// After:
#define TH_GENERIC_FILE "torch/src/generic/Tensor.cpp"
#include "TH/THGenerate<Type>Type.h"
```

Including the header file on the second line has the side effect of including the source code in `Tensor.cpp` with some additional context defined. Let's take a look at one of the headers:

2번째 header file을 including 하는 것에 대한 부수효과는 `Tensor.cpp` 에 대한 내용을 불러오는 것입니다. 

```
#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define real float
#define accreal double
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real Float
#define THInf FLT_MAX
#define TH_REAL_IS_FLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_FLOAT
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
```
What this is doing is bringing in the code from the generic `Tensor.cpp` file and surrounding it with the following macro definitions. For example, we define real as a float, so any code in the generic Tensor implementation that refers to something as a real will have that real replaced with a float. In the corresponding file `THGenerateIntType.h`, the same macro would replace `real` with `int`. 

이것이하는 일은 일반적인`Tensor.cpp` 파일의 코드를 가져 와서 다음 매크로 정의로 둘러 쌉니다. 예를 들어 real을 float으로 정의하므로 generic Tensor 구현에서의 어떤 real로 언급되어 있는 것을 모두 flaot으로 바꿉니다. `THGenerateIntType.h`에서 동일한 매크로가`real`을`int`로 대체합니다.

These output files are returned from `split_types` and added to the list of source files, so we can see how the `.cpp` code for different types is created.

이 출력파일은 `split_types`에서 반환되어 soruce file의 리스트에 추가되므로 다른 유형의`.cpp` 코드가 어떻게 생성되는지 볼 수 있습니다.

There are a few things to note here: First, the `split_types` function is not strictly necessary. We could wrap the code in `Tensor.cpp` in a single file, repeating it for each type. The reason we split the code into separate files is to speed up compilation. Second, what we mean when we talk about the type replacement (e.g. replace real with a float) is that the C preprocessor will perform these subsitutions during compilaiton. Merely surrounding the source code with these macros has no side effects until preprocessing. 

여기에 몇가지 주의할 점이 있습니다. 첫째로 `split_types` 함수는 꼭 필요한 것이 아닙니다. 우리는 `Tensor.cpp`를 하나의 파일로 wrap하여 각 type에 대해 반복 할 수 있습니다. 하지만 분리한 이유는 컴파일 속도를 높이기 위해서입니다. 둘째로 type replacement를 이야기 할때 C 전처리기가 이러한 것들을 수행한다는 것입니다. 이러한 매크로로 소스 코드를 둘러싼 것은 전처리 할 때까지 부작용이 없습니다.

#### Generic Builds (Part Two)
Now that we have source files for all the Tensor types, we need to consider how the corresponding header declarations are created, and also how the conversions from `THTensor_(method)` and `THPTensor_(method)` to `TH<Type>Tensor_method` and `THP<Type>Tensor_method` work. For example, `csrc/generic/Tensor.h` has declarations like:

이제 모든 Tensor type에 대한 source file을 가지고 있으므로 어떻게 대응하는 header 선언이 생성되는 방법을 알야아 합니다. 또한 
 `THTensor_ (method)`및 `THPTensor_ (method)`에서 `TH <Type> Tensor_method`로의 변환을 이해 해야 합니다. 예를들어 `csrc/generic/Tensor.h`은 다음과 같은 선언을 가지고 있습니다.
 
```
THP_API PyObject * THPTensor_(New)(THTensor *ptr);
```

We use the same strategy for generating code in the source files for the headers. In `csrc/Tensor.h`, we do the following:


우리는 헤더에 대한 소스 파일에서 코드를 생성하는 데 동일한 전략을 사용합니다. `csrc / Tensor.h`에서 우리는 다음을 수행합니다 :

```
#include "generic/Tensor.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/Tensor.h"
#include <TH/THGenerateHalfType.h>
```
This has the same effect, where we draw in the code from the generic header, wrapped with the same macro definitions, for each type. The only difference is that the resulting code is contained all within the same header file, as opposed to being split into multiple source files.

이것은 generic header에서 코드를 생성하는 것과 같은 효과를 지니고 있습니다. 각 type에 대해 같은 macro 정의로 의해 wrapped 되어집니다. 유일한 차이점은 결과 코드가 여러 소스 파일로 분할되지 않고 동일한 헤더 파일에 모두 포함되어 있다는 것입니다.

Lastly, we need to consider how we "convert" or "substitute" the function types. If we look in the same header file, we see a bunch of `#define` statements, including:

마지막으로 함수 유형을 "변환"하거나 "대체"하는 방법을 고려해야합니다. 같은 헤더 파일을 보면, 다음과 같은`#define` 명령문이 많이 있습니다.

```
#define THPTensor_(NAME)            TH_CONCAT_4(THP,Real,Tensor_,NAME)
```
This macro says that any string in the source code matching the format `THPTensor_(NAME)` should be replaced with `THPRealTensor_NAME`, where Real is derived from whatever the symbol Real is `#define`'d to be at the time. Because our header code and source code is surrounded by macro definitions for all the types as seen above, after the preprocessor has run, the resulting code is what we would expect. The code in the `TH` library defines the same macro for `THTensor_(NAME)`, supporting the translation of those functions as well. In this way, we end up with header and source files with specialized code.

이 매크로는 'THPTensor_ (NAME)'형식과 일치하는 소스 코드의 문자열은 THPRealTensor_NAME으로 대체되어야한다고 말합니다. 여기서 Real 이라는 것은 Real 이라는 것으로 `#define` 에 의해 전처리 된 것을 말합니다. 헤더코드와 소스코드는 위에서 보았듯이 macro 정의로 쌓여져 있습니다. 전처리기가 실행된 이후에 모든 결과코드는 우리가 기대한 것입니다. `TH` 라이브러리의 코드는 `THTensor_(NAME)`와 동일한 macro를 정의합니다. 이러한 것은 이들 함수를 적절하게 변환합니다. 이러한 방법으로 헤더와 소스파일을 specialized code로 바꿉니다.


####Module Objects and Type Methods
Now we have seen how we have wrapped `TH`'s Tensor definition in `THP`, and generated THP methods such as `THPFloatTensor_init(...)`. Now we can explore what the above code actually does in terms of the module we are creating. The key line in `THPTensor_(init)` is:

이제 우리는 `THP`에 정의된 `TH` 에 대한 wrapped 를 가지고 있습니다. 또한 `THPFloatTensor_init(...)`와 같은 THP 메소드를 가지고 있습니다. 이제 우리는 위에서 작성한 코드가 실제로 무슨 일을 하는지를 알아볼 것입니다.


```
# THPTensorBaseStr, THPTensorType are also macros that are specific 
# to each type
PyModule_AddObject(module, THPTensorBaseStr, (PyObject *)&THPTensorType);
```
This function registers our Tensor objects to the extension module, so we can use THPFloatTensor, THPIntTensor, etc. in our Python code.

이 함수는 Tensor object를 extension module에 등록합니다. 이제 우리는 THPFloatTensor, THPIntTensor 를 Python 코드에서 사용할 수 있습니다.

Just being able to create Tensors isn't very useful - we need to be able to call all the methods that `TH` defines. A simple example shows calling the in-place `zero_` method on a Tensor.

단지 Tensor를 생성할 수 있다는 것은 유용하지 않습니다. 우리는 `TH` 에서 정의된 모든 method를 호출 할 수 있어야 합니다. 예를 들어 Tensor의 `zero_` method를 호출하는 것을 보여줍니다.


```
x = torch.FloatTensor(10)
x.zero_()
```
Let's start by seeing how we add methods to newly defined types. One of the fields in the "type object" is `tp_methods`. This field holds an array of method definitions (`PyMethodDef`s) and is used to associate methods (and their underlying C/C++ implementations) with a type. Suppose we wanted to define a new method on our `PyFloatObject` that replaces the value. We could implement this as follows:

새로 정의된 type에 method를 정의하는 방법부터 살펴봅니다. "type object" 의 필드중 하나는 `tp_methods` 입니다. 이 field는 메소드 정의 (`PyMethodDef`s)의 배열을 담고 있으며 메소드와 그 하부에있는 C / C ++ 구현을 하나의 타입과 연관 시키는데 사용됩니다.`PyFloatObject`에 새로운 값을 대체하는 메소드를 정의하고 싶다고합시다. 다음과 같이 구현할 수 있습니다.

```
static PyObject * replace(PyFloatObject *self, PyObject *args) {
	double val;
	if (!PyArg_ParseTuple(args, "d", &val))
		return NULL;
	self->ob_fval = val;
	Py_RETURN_NONE
}
```
This is equivalent to the Python method:
```
def replace(self, val):
	self.ob_fval = fal
```
It is instructive to read more about how defining methods works in CPython. In general, methods take as the first parameter the instance of the object, and optionally parameters for the positional arguments and keyword arguments. This static function is registered as a method on our float:


CPython에서 메소드 정의가 어떻게 작동하는지 더 자세히 읽어 보는 것이 좋습니다. 일반적으로 메서드는 첫 번째 매개 변수로 개체 인스턴스를 가져오고 선택적으로 위치 인수 및 키워드 인수에 대한 매개 변수를 사용합니다. 이 static 함수는 float에 메서드로 등록됩니다.

```
static PyMethodDef float_methods[] = {
	{"replace", (PyCFunction)replace, METH_VARARGS,
	"replace the value in the float"
	},
	{NULL} /* Sentinel */
}
```


This registers a method called replace, which is implemented by the C function of the same name. The `METH_VARARGS` flag indicates that the method takes a tuple of arguments representing all the arguments to the function. This array is set to the `tp_methods` field of the type object, and then we can use the `replace` method on objects of that type.

이것은 같은 이름의 C 로 구현된 replace 라는 함수를 등록합니다. `METH_VARARGS`라는 플래그는 메소드가 함수의 모든 인자를 나타내는 일종의 튜플을 나타냅니다. 이 배열은 `tp_methods` 의 필드로 설정이 되며 우리는 그 타입의 객체들에 대해서 `replace` 메소드를 사용 할 수 있습니다.

We would like to be able to call all of the methods for `TH` tensors on our `THP` tensor equivalents. However, writing wrappers for all of the `TH` methods would be time-consuming and error prone. We need a better way to do this.

이제 우리는 `THP` 텐서와 같은 `TH` 에서 모든 메소드를 호출하기를 원합니다. 그러나 모든 'TH'방법에 대한 래퍼 작성은 시간이 오래 걸리며 오류가 발생하기 쉽습니다. 우리는 더 나은 방법이 필요합니다.

#### PyTorch cwrap
PyTorch implements its own cwrap tool to wrap the `TH` Tensor methods for use in the Python backend. We define a `.cwrap` file containing a series of C method declarations in our custom YAML format (http://yaml.org). The cwrap tool takes this file and outputs `.cpp` source files containing the wrapped methods in a format that is compatible with our `THPTensor` Python object and the Python C extension method calling format. This tool is used to generate code to wrap not only `TH`, but also `CuDNN`. It is defined to be extensible.

Pytorch 는 Python backend에서 사용하기 위한 `TH` Tensor method를 wrap 하기 위한 cwrap tool을 구현합니다. YAML 포맷 (http://yaml.org)에서 일련의 C 메소드 선언을 포함하는`.cwrap` 파일을 정의합니다. cwrap tool은 이 파일을 가져와서 `THPTesnor` 와 Python object, Python C extenstion method와 호환 가능한 `.cpp` 소스파일을 출력합니다. 이 tool은 'TH'뿐만 아니라 'CuDNN'을 감싸는 코드를 생성하는 데 사용됩니다. 그것은 확장 가능하도록 정의됩니다.

An example YAML "declaration" for the in-place `addmv_` function is as follows:
```
[[
  name: addmv_
  cname: addmv
  return: self
  arguments:
    - THTensor* self
    - arg: real beta
      default: AS_REAL(1)
    - THTensor* self
    - arg: real alpha
      default: AS_REAL(1)
    - THTensor* mat
    - THTensor* vec
]]
```
The architecture of the cwrap tool is very simple. It reads in a file, and then processes it with a series of **plugins.** See `tools/cwrap/plugins/__init__.py` for documentation on all the ways a plugin can alter the code.


cwrap 도구의 아키텍처는 매우 간단합니다. 파일을 읽어 들여 일련의 **plugins.**으로 처리합니다 **plugins.**이 코드를 변경할 수있는 모든 방법에 대한 문서는`tools / cwrap / plugins / __ init __. py`를보십시오.

The source code generation occurs in a series of passes. First, the YAML "declaration" is parsed and processed. Then the source code is generated piece-by-piece - adding things like argument checks and extractions, defining the method header, and the actual call to the underlying library such as `TH`. Finally, the cwrap tool allows for processing the entire file at a time. The resulting output for `addmv_` can be explored here: https://gist.github.com/killeent/c00de46c2a896335a52552604cc4d74b.

소스 코드 생성은 일련의 과정에서 발생합니다. 먼저 YAML "declaration" 이 파싱되어 처리됩니다. 그런 다음 소스 코드가 piece-by-piece로 생성됩니다. 인자 추출, 확인, 메소드 헤더 정의, `TH` 과 같은 라이브러리 호출을 추가합니다. cwrap tool은 전체 파일에 대한 처리를 한번에 합니다. `addmv_` 에 대한 결과는 다음에서 볼 수 있습니다.


In order to interface with the CPython backend, the tool generates an array of `PyMethodDef`s that can be stored or appended to the `THPTensor`'s `tp_methods` field.

CPython backend와 interface하기 위해 이 tool은 `THPTensor`의`tp_methods` 필드에 저장하거나 추가 할 수있는`PyMethodDef` 배열을 생성합니다


In the specific case of wrapping Tensor methods, the build process first generates the output source file from `TensorMethods.cwrap`. This source file is `#include`'d in the generic Tensor source file. This all occurs before the preprocessor does its magic. As a result, all of the method wrappers that are generated undergo the same pass as the `THPTensor` code above. Thus a single generic declaration and definition is specialized for each type as well.

Tensor method를 specific case로 wrapping 하기 위해서 빌드 프로세스는 먼저 `TensorMethods.cwrap`에서 출력 원본 파일을 생성합니다. 이 소스 파일은 generic Tensor 소스 파일에`#include `되어 있습니다. 이 모든 것은 전처리 기가 마술을하기 전에 발생합니다. 결과적으로, 생성 된 모든 메소드 랩퍼는 위의`THPTensor` 코드와 동일한 단계를 거칩니다. 따라서 단일 일반 선언 및 정의는 각 유형에 대해서도 특수화됩니다.


#### Putting It All Together

So far, we have shown how we extend the Python interpreter to create a new extension module, how such a module defines our new `THPTensor` type, and how we can generate source code for Tensors of all types that interface with `TH`. Briefly, we will touch on compilation.


지금까지 파이썬 인터프리터를 확장하여 새로운 확장 모듈을 만드는 방법, 새로운 모듈의 THPTensor 유형을 정의하는 방법, TH로 인터페이스하는 모든 유형의 Tensors에 대한 소스 코드를 생성하는 방법을 살펴 보았습니다. 

Setuptools allows us to define an Extension for compilation. The entire `torch._C` extension is compiled by collecting all of the source files, header files, libraries, etc. and creating a setuptools `Extension`. Then setuptools handles building the extension itself. I will explore the build process more in a subsequent post.

Setuptools는 컴파일을위한 Extension을 정의 할 수있게 해줍니다. 전체`torch._C `확장은 모든 소스 파일, 헤더 파일, 라이브러리 등을 수집하고 setuptools`Extension`을 생성하여 컴파일됩니다. 그런 다음 setuptools가 확장 기능 자체를 처리합니다. 후속 게시물에서 빌드 프로세스를 더 자세히 살펴 보겠습니다.

To summarize, let's revisit our four questions:

요약하기 위해 네가지 질문을 다시 살펴보겠습니다.

- How does PyTorch extend the Python interpreter to define a Tensor type that can be manipulated from Python code?

It uses CPython's framework for extending the Python interpreter and defining new types, while taking special care to generate code for all types.

- PyTorch는 Python 코드에서 조작 할 수있는 Tensor 유형을 정의하기 위해 Python 인터프리터를 어떻게 확장합니까?

모든 type에 대한 specialize 를 고려하면서 새로운 유형을 정의하고 Python interpreter를 확장하기 위해 CPython 프레임워크를 사용합니다.

- How does PyTorch wrap the C libraries that actually define the Tensor's properties and methods?

It does so by defining a new type, `THPTensor`, that is backed by a `TH` Tensor. Function calls are forwarded to this tensor via the CPython backend's conventions.

- PyTorch는 실제로 Tensor의 속성과 메서드를 정의하는 C 라이브러리를 어떻게 래핑합니까?

그것은 새로운 타입 인 `THPTensor`를 정의합니다. 이는 `TH` Tensor의 뒷단이 됩니다. 함수 호출은 CPython 백엔드의 규칙을 통해이 텐서에 전달됩니다.

- How does PyTorch cwrap work to generate code for Tensor methods?

It takes our custom YAML-formatted code and generates source code for each method by processing it through a series of steps using a number of plugins.

- PyTorch는 Tensor 메서드에 대한 코드를 생성하기 위해 어떻게 작동합니까?

사용자 정의 YAML 형식의 코드를 사용하고 여러 플러그인을 사용하여 일련의 단계로 처리하여 각 메소드의 소스 코드를 생성합니다.

- How does PyTorch's build system take all of these components to compile and generate a workable application?

It takes a bunch of source/header files, libraries, and compilation directives to build an extension using Setuptools.

This is just a snapshot of parts of the build system for PyTorch. There is more nuance, and detail, but I hope this serves as a gentle introduction to a lot of the components of our Tensor library.

- PyTorch의 빌드 시스템은 이러한 모든 구성 요소를 사용하여 실행 가능한 응용 프로그램을 컴파일하고 생성하는 방법은 무엇입니까?

Setuptools를 사용하여 extension을 빌드하기 위해 많은 소스 / 헤더 파일, 라이브러리 및 컴파일 지시문이 필요합니다.

이것은 PyTorch 빌드 시스템의 일부분을 보여주는 스냅 샷입니다. 더 많은 뉘앙스와 세부 사항이 있습니다 만, 이것이 Tensor 라이브러리의 많은 구성 요소에 대한 부드러운 소개 역할을하기를 바랍니다.

#### Resources:

 - https://docs.python.org/3.7/extending/index.html is invaluable for understanding how to write C/C++ Extension to Python