设计模式概述
设计模式（Design Patterns） 是软件开发人员在软件开发过程中面临的一般问题的解决方案。这些解决方案是众多软件开发人员经过相当长的一段时间的试验和错误总结出来的。

目的（Purpose）是提高代码的可重用性、使代码更容易被他人理解、保证代码可靠性以及降低程序维护的复杂度。

创建型模式（Creational Patterns）：这些设计模式提供了一种在创建对象的同时隐藏创建逻辑的方式，而不是使用 new 运算符直接实例化对象。

结构型模式（Structural Patterns）：这些设计模式关注类和对象的组合。继承在这里不大被考虑，而更偏向于类的关联。结构型类模式采用继承以外的机制来组合对象，从而获得更大的灵活性。

行为型模式（Behavioral Patterns）：这些设计模式特别关注对象之间的通信。

类型	设计模型	描述
创建型

工厂方法模式（Factory Method Pattern）	定义一个创建对象的接口，但由子类决定实例化的类是哪一个。
抽象工厂模式（Abstract Factory Pattern）	提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们的具体类。
建造者模式（Builder Pattern）	将一个复杂对象的构建与其表示分离，使同样的构建过程可以创建不同的表示。
原型模式（Prototype Pattern）	用原型实例指定创建对象的种类，并通过拷贝这些原型创建新的对象。
单例模式（Singleton Pattern）	确保一个类只有一个实例，并提供一个全局访问点。
结构型	适配器模式（Adapter Pattern）	将一个类的接口转换成客户希望的另外一个接口。
桥接模式（Bridge Pattern）	将抽象部分与实现部分分离，使它们都可以独立变化。
组合模式（Composite Pattern）	将对象组合成树形结构以表示“部分-整体”的层次结构。
装饰器模式（Decorator Pattern）	动态地给一个对象添加一些额外的职责，就增加功能来说，装饰模式比生成子类更加灵活。
外观模式（Facade Pattern）	为子系统中的一组接口提供一个统一的接口，外观模式定义了一个高层接口，使得子系统更容易使用。
享元模式（Flyweight Pattern）	运用共享技术来有效地支持大量细粒度对象的复用。
代理模式（Proxy Pattern）	为其他对象提供一种代理以控制对这个对象的访问。
行为型	责任链模式（Chain of Responsibility Pattern）	将请求的发送者和接收者解耦，让多个对象都有机会处理这个请求。
命令模式（Command Pattern）	将请求封装成对象，从而允许使用不同的请求、队列或日志请求来参数化其他对象。
解释器模式（Interpreter Pattern）	给定一个语言，定义它的文法的一种表示，并定义一个解释器，该解释器使用该表示来解释语言中的句子。
迭代器模式（Iterator Pattern）	提供一种方法顺序访问一个聚合对象中的各个元素，而又不需暴露该对象的内部表示。
中介者模式（Mediator Pattern）	用一个中介对象来封装一系列的对象交互，中介者使各对象不需要显示地相互引用。
备忘录模式（Memento Pattern）	在不破坏封装的情况下，捕获一个对象的内部状态，并在该对象之外保存这个状态。
观察者模式（Observer Pattern）	定义对象间的一种一对多的依赖关系，当一个对象状态发生改变时，所有依赖于它的对象都得到通知并自动更新。
状态模式（State Pattern）	允许一个对象在其内部状态改变时改变它的行为。
策略模式（Strategy Pattern）	
定义一系列算法，将每个算法封装起来，并使它们可以互换。

模板方法模式（Template Method Pattern）	定义一个操作中的算法的框架，而将一些步骤延迟到子类中。
访问者模式（Visitor Pattern）	表示一个作用于某对象结构中的各元素的操作，可以在不改变各元素类的前提下定义作用于这些元素的新操作。
创建型
这些设计模式提供了一种在创建对象的同时隐藏创建逻辑的方式，而不是使用 new 运算符直接实例化对象。

1.工厂方法模式（Factory Method Pattern）：
说明：定义一个创建对象的接口，但由子类决定实例化的类是哪一个。
应用：当一个类不知道它所必须创建的对象的类时。
示例：
#include <iostream>
 
// 抽象产品类
class Product {
public:
    virtual void operation() = 0;
};
 
// 具体产品类 A
class ConcreteProductA : public Product {
public:
    void operation() override {
        std::cout << "ConcreteProductA operation\n";
    }
};
 
// 具体产品类 B
class ConcreteProductB : public Product {
public:
    void operation() override {
        std::cout << "ConcreteProductB operation\n";
    }
};
 
// 抽象工厂类
class Factory {
public:
    virtual Product* createProduct() = 0;
};
 
// 具体工厂类 A
class ConcreteFactoryA : public Factory {
public:
    Product* createProduct() override {
        return new ConcreteProductA();
    }
};
 
// 具体工厂类 B
class ConcreteFactoryB : public Factory {
public:
    Product* createProduct() override {
        return new ConcreteProductB();
    }
};
 
int main() {
    Factory* factoryA = new ConcreteFactoryA();
    Product* productA = factoryA->createProduct();
    productA->operation();
 
    Factory* factoryB = new ConcreteFactoryB();
    Product* productB = factoryB->createProduct();
    productB->operation();
 
    delete factoryA;
    delete productA;
    delete factoryB;
    delete productB;
 
    return 0;
}
2.抽象工厂模式（Abstract Factory Pattern）：
说明：提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们的具体类。
应用：当系统需要独立于它的产品的创建、组合和表示时。
示例：
#include <iostream>
 
// 抽象产品类 A
class AbstractProductA {
public:
    virtual void operationA() = 0;
};
 
// 具体产品类 A1
class ProductA1 : public AbstractProductA {
public:
    void operationA() override {
        std::cout << "ProductA1 operationA\n";
    }
};
 
// 具体产品类 A2
class ProductA2 : public AbstractProductA {
public:
    void operationA() override {
        std::cout << "ProductA2 operationA\n";
    }
};
 
// 抽象产品类 B
class AbstractProductB {
public:
    virtual void operationB() = 0;
};
 
// 具体产品类 B1
class ProductB1 : public AbstractProductB {
public:
    void operationB() override {
        std::cout << "ProductB1 operationB\n";
    }
};
 
// 具体产品类 B2
class ProductB2 : public AbstractProductB {
public:
    void operationB() override {
        std::cout << "ProductB2 operationB\n";
    }
};
 
// 抽象工厂类
class AbstractFactory {
public:
    virtual AbstractProductA* createProductA() = 0;
    virtual AbstractProductB* createProductB() = 0;
};
 
// 具体工厂类 1
class ConcreteFactory1 : public AbstractFactory {
public:
    AbstractProductA* createProductA() override {
        return new ProductA1();
    }
 
    AbstractProductB* createProductB() override {
        return new ProductB1();
    }
};
 
// 具体工厂类 2
class ConcreteFactory2 : public AbstractFactory {
public:
    AbstractProductA* createProductA() override {
        return new ProductA2();
    }
 
    AbstractProductB* createProductB() override {
        return new ProductB2();
    }
};
 
int main() {
    AbstractFactory* factory1 = new ConcreteFactory1();
    AbstractProductA* productA1 = factory1->createProductA();
    AbstractProductB* productB1 = factory1->createProductB();
    productA1->operationA();
    productB1->operationB();
 
    AbstractFactory* factory2 = new ConcreteFactory2();
    AbstractProductA* productA2 = factory2->createProductA();
    AbstractProductB* productB2 = factory2->createProductB();
    productA2->operationA();
    productB2->operationB();
 
    delete factory1;
    delete productA1;
    delete productB1;
    delete factory2;
    delete productA2;
    delete productB2;
 
    return 0;
}
3.建造者模式（Builder Pattern）：
说明：将一个复杂对象的构建与其表示分离，使同样的构建过程可以创建不同的表示。
应用：当对象的创建算法应该独立于该对象的组成部分以及它们的装配方式时。
示例：
#include <iostream>
#include <string>
 
// 产品类
class Product {
public:
    void setPartA(const std::string& partA) {
        partA_ = partA;
    }
 
    void setPartB(const std::string& partB) {
        partB_ = partB;
    }
 
    void setPartC(const std::string& partC) {
        partC_ = partC;
    }
 
    void show() {
        std::cout << "Part A: " << partA_ << std::endl;
        std::cout << "Part B: " << partB_ << std::endl;
        std::cout << "Part C: " << partC_ << std::endl;
    }
 
private:
    std::string partA_;
    std::string partB_;
    std::string partC_;
};
 
// 抽象建造者类
class Builder {
public:
    virtual void buildPartA() = 0;
    virtual void buildPartB() = 0;
    virtual void buildPartC() = 0;
    virtual Product* getResult() = 0;
};
 
// 具体建造者类
class ConcreteBuilder : public Builder {
public:
    ConcreteBuilder() {
        product_ = new Product();
    }
 
    void buildPartA() override {
        product_->setPartA("PartA");
    }
 
    void buildPartB() override {
        product_->setPartB("PartB");
    }
 
    void buildPartC() override {
        product_->setPartC("PartC");
    }
 
    Product* getResult() override {
        return product_;
    }
 
private:
    Product* product_;
};
 
// 导演类
class Director {
public:
    void construct(Builder* builder) {
        builder->buildPartA();
        builder->buildPartB();
        builder->buildPartC();
    }
};
 
int main() {
    Director director;
    ConcreteBuilder builder;
    director.construct(&builder);
 
    Product* product = builder.getResult();
    product->show();
 
    delete product;
 
    return 0;
}
4.原型模式（Prototype Pattern）：
说明：用原型实例指定创建对象的种类，并通过拷贝这些原型创建新的对象。（即通过复制现有实例来创建新的实例，无需知道相应类的信息。）
应用：当要实例化的类是在运行时刻指定时，例如通过动态装载。
示例：
#include <iostream>
#include <string>
 
// 原型基类
class Prototype {
public:
    virtual Prototype* clone() = 0;
    virtual void operation() = 0;
};
 
// 具体原型类
class ConcretePrototype : public Prototype {
public:
    ConcretePrototype(int id, const std::string& name) : id_(id), name_(name) {}
 
    Prototype* clone() override {
        return new ConcretePrototype(*this);
    }
 
    void operation() override {
        std::cout << "ID: " << id_ << ", Name: " << name_ << std::endl;
    }
 
private:
    int id_;
    std::string name_;
};
 
int main() {
    ConcretePrototype prototype(1, "Prototype");
    Prototype* clonedPrototype = prototype.clone();
    clonedPrototype->operation();
 
    delete clonedPrototype;
 
    return 0;
}
5.单例模式（Singleton Pattern）：
说明：确保一个类只有一个实例，并提供一个全局访问点。
应用：当类只能有一个实例而且客户可以从一个众所周知的访问点访问它时。
示例：
#include <iostream>
 
// 单例类
class Singleton {
public:
    // 获取单例实例的静态方法
    static Singleton& getInstance() {
        static Singleton instance; // 线程安全的局部静态变量
        return instance;
    }
 
    // 示例方法
    void showMessage() const {
        std::cout << "Hello, Singleton!" << std::endl;
    }
 
private:
    // 私有构造函数
    Singleton() {}
 
    // 禁止拷贝构造函数和赋值操作符
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};
 
int main() {
    // 获取单例实例并调用示例方法
    Singleton& singleton = Singleton::getInstance();
    singleton.showMessage();
 
    return 0;
}
结构型
这些设计模式关注类和对象的组合。继承在这里不大被考虑，而更偏向于类的关联。结构型类模式采用继承以外的机制来组合对象，从而获得更大的灵活性。

6.适配器模式（Adapter Pattern）：
说明：将一个类的接口转换成客户希望的另外一个接口。
应用：当想要使用一个已经存在的类，但它的接口不符合你的要求时。
示例：
#include <iostream>
#include <string>
 
// 目标接口
class Target {
public:
    virtual void request() = 0;
};
 
// 需要适配的类
class Adaptee {
public:
    void specificRequest(const std::string& message) {
        std::cout << "Adaptee: " << message << std::endl;
    }
};
 
// 类适配器
class Adapter : public Target, private Adaptee {
public:
    void request() override {
        specificRequest("Adapter pattern");
    }
};
 
int main() {
    // 使用适配器
    Target* target = new Adapter();
    target->request();
 
    delete target;
 
    return 0;
}
7.桥接模式（Bridge Pattern）：
说明：将抽象部分与实现部分分离，使它们都可以独立变化。
应用：当希望一个抽象和它的实现部分可以独立地变化时。
示例：
#include <iostream>
 
// 实现部分的接口
class Implementor {
public:
    virtual void operationImpl() = 0;
};
 
// 具体实现类 A
class ConcreteImplementorA : public Implementor {
public:
    void operationImpl() override {
        std::cout << "ConcreteImplementorA operationImpl\n";
    }
};
 
// 具体实现类 B
class ConcreteImplementorB : public Implementor {
public:
    void operationImpl() override {
        std::cout << "ConcreteImplementorB operationImpl\n";
    }
};
 
// 抽象部分
class Abstraction {
public:
    Abstraction(Implementor* impl) : implementor_(impl) {}
 
    virtual void operation() = 0;
 
protected:
    Implementor* implementor_;
};
 
// 扩展抽象部分 A
class RefinedAbstractionA : public Abstraction {
public:
    RefinedAbstractionA(Implementor* impl) : Abstraction(impl) {}
 
    void operation() override {
        std::cout << "RefinedAbstractionA operation ";
        implementor_->operationImpl();
    }
};
 
// 扩展抽象部分 B
class RefinedAbstractionB : public Abstraction {
public:
    RefinedAbstractionB(Implementor* impl) : Abstraction(impl) {}
 
    void operation() override {
        std::cout << "RefinedAbstractionB operation ";
        implementor_->operationImpl();
    }
};
 
int main() {
    Implementor* implA = new ConcreteImplementorA();
    Abstraction* abstractionA = new RefinedAbstractionA(implA);
    abstractionA->operation();
 
    Implementor* implB = new ConcreteImplementorB();
    Abstraction* abstractionB = new RefinedAbstractionB(implB);
    abstractionB->operation();
 
    delete implA;
    delete abstractionA;
    delete implB;
    delete abstractionB;
 
    return 0;
}
8.组合模式（Composite Pattern）：
说明：将对象组合成树形结构以表示“部分-整体”的层次结构。
应用：当希望用户忽略组合对象与单个对象的区别时。
示例：
#include <iostream>
#include <vector>
 
// 抽象构件
class Component {
public:
    virtual void operation() = 0;
};
 
// 叶子构件
class Leaf : public Component {
public:
    void operation() override {
        std::cout << "Leaf operation\n";
    }
};
 
// 容器构件
class Composite : public Component {
public:
    void add(Component* component) {
        children_.push_back(component);
    }
 
    void operation() override {
        std::cout << "Composite operation\n";
        for (Component* component : children_) {
            component->operation();
        }
    }
 
private:
    std::vector<Component*> children_;
};
 
int main() {
    Leaf* leaf1 = new Leaf();
    Leaf* leaf2 = new Leaf();
    Leaf* leaf3 = new Leaf();
 
    Composite* composite = new Composite();
    composite->add(leaf1);
    composite->add(leaf2);
 
    Composite* composite2 = new Composite();
    composite2->add(leaf3);
    composite2->add(composite);
 
    composite2->operation();
 
    delete leaf1;
    delete leaf2;
    delete leaf3;
    delete composite;
    delete composite2;
 
    return 0;
}
9.装饰器模式（Decorator Pattern）：
说明：动态地给一个对象添加一些额外的职责，就增加功能来说，装饰模式比生成子类更加灵活。
应用：当给一个对象添加功能，但是又不想生成一个子类时。
示例：
#include <iostream>
#include <memory>
 
// 抽象构件
class Component {
public:
    virtual void operation() = 0;
};
 
// 具体构件
class ConcreteComponent : public Component {
public:
    void operation() override {
        std::cout << "ConcreteComponent operation\n";
    }
};
 
// 抽象装饰类
class Decorator : public Component {
public:
    Decorator(Component* component) : component_(component) {}
 
    void operation() override {
        if (component_ != nullptr) {
            component_->operation();
        }
    }
 
protected:
    Component* component_;
};
 
// 具体装饰类 A
class ConcreteDecoratorA : public Decorator {
public:
    ConcreteDecoratorA(Component* component) : Decorator(component) {}
 
    void operation() override {
        Decorator::operation();
        addedBehavior();
    }
 
    void addedBehavior() {
        std::cout << "ConcreteDecoratorA addedBehavior\n";
    }
};
 
// 具体装饰类 B
class ConcreteDecoratorB : public Decorator {
public:
    ConcreteDecoratorB(Component* component) : Decorator(component) {}
 
    void operation() override {
        Decorator::operation();
        addedBehavior();
    }
 
    void addedBehavior() {
        std::cout << "ConcreteDecoratorB addedBehavior\n";
    }
};
 
int main() {
    Component* component = new ConcreteComponent();
    Component* decoratedComponentA = new ConcreteDecoratorA(component);
    Component* decoratedComponentB = new ConcreteDecoratorB(decoratedComponentA);
 
    decoratedComponentB->operation();
 
    delete decoratedComponentB; // Deleting the decorator will automatically delete the entire decoration chain
    // No need to delete component or decoratedComponentA, as they are not owned by the client
 
    return 0;
}
10.外观模式（Facade Pattern）：
说明：为子系统中的一组接口提供一个统一的接口，外观模式定义了一个高层接口，使得子系统更容易使用。
应用：当要为一个复杂的子系统提供一个简单接口时。
示例：
#include <iostream>
 
// 子系统 A
class SubsystemA {
public:
    void operationA() {
        std::cout << "SubsystemA operation\n";
    }
};
 
// 子系统 B
class SubsystemB {
public:
    void operationB() {
        std::cout << "SubsystemB operation\n";
    }
};
 
// 外观类
class Facade {
public:
    Facade() : subsystemA_(), subsystemB_() {}
 
    void operation() {
        subsystemA_.operationA();
        subsystemB_.operationB();
    }
 
private:
    SubsystemA subsystemA_;
    SubsystemB subsystemB_;
};
 
int main() {
    Facade facade;
    facade.operation();
 
    return 0;
}
11.享元模式（Flyweight Pattern）：
说明：运用共享技术来有效地支持大量细粒度对象的复用。
应用：当应用程序使用大量相似对象，造成很大的存储开销时。
示例：
#include <iostream>
#include <string>
#include <map>
 
// 享元接口
class Flyweight {
public:
    virtual void operation(const std::string& extrinsicState) = 0;
};
 
// 具体享元类
class ConcreteFlyweight : public Flyweight {
public:
    void operation(const std::string& extrinsicState) override {
        std::cout << "ConcreteFlyweight with state: " << extrinsicState << std::endl;
    }
};
 
// 享元工厂
class FlyweightFactory {
public:
    Flyweight* getFlyweight(const std::string& key) {
        if (flyweights_.find(key) == flyweights_.end()) {
            flyweights_[key] = new ConcreteFlyweight();
        }
        return flyweights_[key];
    }
 
private:
    std::map<std::string, Flyweight*> flyweights_;
};
 
int main() {
    FlyweightFactory factory;
    Flyweight* flyweight1 = factory.getFlyweight("key1");
    flyweight1->operation("state1");
 
    Flyweight* flyweight2 = factory.getFlyweight("key2");
    flyweight2->operation("state2");
 
    // Reusing the flyweight
    Flyweight* flyweight3 = factory.getFlyweight("key1");
    flyweight3->operation("state3");
 
    delete flyweight1;
    delete flyweight2;
 
    return 0;
}
12.代理模式（Proxy Pattern）：
说明：为其他对象提供一种代理以控制对这个对象的访问。
应用：当需要用比较通用和复杂的对象代替简单的对象时。
示例：
#include <iostream>
 
// 抽象主题
class Subject {
public:
    virtual void request() = 0;
};
 
// 具体主题
class RealSubject : public Subject {
public:
    void request() override {
        std::cout << "RealSubject request\n";
    }
};
 
// 代理类
class Proxy : public Subject {
public:
    Proxy(Subject* realSubject) : realSubject_(realSubject) {}
 
    void request() override {
        // Performing additional operations before forwarding the request to the real subject
        std::cout << "Proxy request: ";
        realSubject_->request();
    }
 
private:
    Subject* realSubject_;
};
 
int main() {
    RealSubject* realSubject = new RealSubject();
    Proxy* proxy = new Proxy(realSubject);
 
    proxy->request();
 
    delete proxy;
    delete realSubject;
 
    return 0;
}
行为型
这些设计模式特别关注对象之间的通信。

13.责任链模式（Chain of Responsibility Pattern）：
说明：将请求的发送者和接收者解耦，让多个对象都有机会处理这个请求。
应用：当要给多个对象处理请求的机会，而不明确指定处理请求的对象时。
示例：
#include <iostream>
#include <string>
 
// 抽象处理者
class Handler {
public:
    Handler(Handler* successor = nullptr) : successor_(successor) {}
 
    virtual void handleRequest(const std::string& request) {
        if (successor_ != nullptr) {
            successor_->handleRequest(request);
        }
    }
 
protected:
    Handler* successor_;
};
 
// 具体处理者 A
class ConcreteHandlerA : public Handler {
public:
    void handleRequest(const std::string& request) override {
        if (request == "A") {
            std::cout << "ConcreteHandlerA handles the request\n";
        } else {
            Handler::handleRequest(request);
        }
    }
};
 
// 具体处理者 B
class ConcreteHandlerB : public Handler {
public:
    void handleRequest(const std::string& request) override {
        if (request == "B") {
            std::cout << "ConcreteHandlerB handles the request\n";
        } else {
            Handler::handleRequest(request);
        }
    }
};
 
// 具体处理者 C
class ConcreteHandlerC : public Handler {
public:
    void handleRequest(const std::string& request) override {
        if (request == "C") {
            std::cout << "ConcreteHandlerC handles the request\n";
        } else {
            Handler::handleRequest(request);
        }
    }
};
 
int main() {
    Handler* handlerA = new ConcreteHandlerA();
    Handler* handlerB = new ConcreteHandlerB(handlerA); // Set successor
    Handler* handlerC = new ConcreteHandlerC(handlerB); // Set successor
 
    handlerC->handleRequest("B");
    handlerC->handleRequest("A");
    handlerC->handleRequest("C");
    handlerC->handleRequest("D"); // No handler can handle this request
 
    delete handlerC;
    delete handlerB;
    delete handlerA;
 
    return 0;
}
14.命令模式（Command Pattern）：
说明：将请求封装成对象，从而允许使用不同的请求、队列或日志请求来参数化其他对象。
应用：当希望系统与其操作都是松耦合的时候。
示例：
#include <iostream>
#include <vector>
 
// 命令接口
class Command {
public:
    virtual void execute() = 0;
};
 
// 具体命令 A
class ConcreteCommandA : public Command {
public:
    void execute() override {
        std::cout << "ConcreteCommandA executed\n";
    }
};
 
// 具体命令 B
class ConcreteCommandB : public Command {
public:
    void execute() override {
        std::cout << "ConcreteCommandB executed\n";
    }
};
 
// 调用者
class Invoker {
public:
    void setCommand(Command* command) {
        command_ = command;
    }
 
    void executeCommand() {
        if (command_ != nullptr) {
            command_->execute();
        }
    }
 
private:
    Command* command_;
};
 
int main() {
    Invoker invoker;
 
    Command* commandA = new ConcreteCommandA();
    invoker.setCommand(commandA);
    invoker.executeCommand();
 
    Command* commandB = new ConcreteCommandB();
    invoker.setCommand(commandB);
    invoker.executeCommand();
 
    delete commandA;
    delete commandB;
 
    return 0;
}
15.解释器模式（Interpreter Pattern）：
说明：给定一个语言，定义它的文法的一种表示，并定义一个解释器，该解释器使用该表示来解释语言中的句子。
应用：当有一个语言需要解释执行，并且你可以将该语言中的句子表示为一个抽象语法树时。
示例：
#include <iostream>
#include <unordered_map>
#include <stack>
 
// 上下文类
class Context {
public:
    void setVariable(const std::string& name, bool value) {
        variables_[name] = value;
    }
 
    bool getVariable(const std::string& name) {
        return variables_[name];
    }
 
private:
    std::unordered_map<std::string, bool> variables_;
};
 
// 抽象表达式类
class AbstractExpression {
public:
    virtual bool interpret(Context& context) = 0;
};
 
// 终结符表达式类
class TerminalExpression : public AbstractExpression {
public:
    TerminalExpression(const std::string& variableName) : variableName_(variableName) {}
 
    bool interpret(Context& context) override {
        return context.getVariable(variableName_);
    }
 
private:
    std::string variableName_;
};
 
// 非终结符表达式类（与操作）
class AndExpression : public AbstractExpression {
public:
    AndExpression(AbstractExpression* expr1, AbstractExpression* expr2)
        : expr1_(expr1), expr2_(expr2) {}
 
    bool interpret(Context& context) override {
        return expr1_->interpret(context) && expr2_->interpret(context);
    }
 
private:
    AbstractExpression* expr1_;
    AbstractExpression* expr2_;
};
 
// 非终结符表达式类（或操作）
class OrExpression : public AbstractExpression {
public:
    OrExpression(AbstractExpression* expr1, AbstractExpression* expr2)
        : expr1_(expr1), expr2_(expr2) {}
 
    bool interpret(Context& context) override {
        return expr1_->interpret(context) || expr2_->interpret(context);
    }
 
private:
    AbstractExpression* expr1_;
    AbstractExpression* expr2_;
};
 
int main() {
    Context context;
    context.setVariable("A", true);
    context.setVariable("B", false);
 
    AbstractExpression* expression =
        new AndExpression(new TerminalExpression("A"), new OrExpression(new TerminalExpression("B"), new TerminalExpression("C")));
 
    bool result = expression->interpret(context);
    std::cout << "Result: " << std::boolalpha << result << std::endl;
 
    delete expression;
 
    return 0;
}
16.迭代器模式（Iterator Pattern）：
说明：提供一种方法顺序访问一个聚合对象中的各个元素，而又不需暴露该对象的内部表示。
应用：当需要提供一种方法顺序访问一个聚合对象中的各个元素，而又不暴露其内部表示时。
示例：
#include <iostream>
#include <vector>
 
// 抽象迭代器
template<typename T>
class Iterator {
public:
    virtual T next() = 0;
    virtual bool hasNext() = 0;
};
 
// 具体迭代器
template<typename T>
class ConcreteIterator : public Iterator<T> {
public:
    ConcreteIterator(const std::vector<T>& data) : data_(data), index_(0) {}
 
    T next() override {
        return data_[index_++];
    }
 
    bool hasNext() override {
        return index_ < data_.size();
    }
 
private:
    std::vector<T> data_;
    size_t index_;
};
 
// 抽象聚合类
template<typename T>
class Aggregate {
public:
    virtual Iterator<T>* createIterator() = 0;
    virtual void add(const T& item) = 0;
    virtual T at(int index) = 0;
    virtual int size() = 0;
};
 
// 具体聚合类
template<typename T>
class ConcreteAggregate : public Aggregate<T> {
public:
    Iterator<T>* createIterator() override {
        return new ConcreteIterator<T>(data_);
    }
 
    void add(const T& item) override {
        data_.push_back(item);
    }
 
    T at(int index) override {
        return data_[index];
    }
 
    int size() override {
        return data_.size();
    }
 
private:
    std::vector<T> data_;
};
 
int main() {
    ConcreteAggregate<int> aggregate;
    aggregate.add(1);
    aggregate.add(2);
    aggregate.add(3);
 
    Iterator<int>* iterator = aggregate.createIterator();
    while (iterator->hasNext()) {
        std::cout << iterator->next() << " ";
    }
    std::cout << std::endl;
 
    delete iterator;
 
    return 0;
}
17.中介者模式（Mediator Pattern）：
说明：用一个中介对象来封装一系列的对象交互。中介者使各对象不需要显示地相互引用，从而使其耦合松散，而且可以独立地改变它们之间的交互。
应用：当多个对象之间存在复杂的交互关系，而这些关系可以被分离处理时。
示例：
#include <iostream>
#include <string>
#include <vector>
 
// 中介者抽象类
class Mediator {
public:
    virtual void sendMessage(const std::string& message, Colleague* colleague) = 0;
};
 
// 抽象同事类
class Colleague {
public:
    Colleague(Mediator* mediator) : mediator_(mediator) {}
 
    virtual void send(const std::string& message) = 0;
    virtual void receive(const std::string& message) = 0;
 
protected:
    Mediator* mediator_;
};
 
// 具体同事类 A
class ConcreteColleagueA : public Colleague {
public:
    ConcreteColleagueA(Mediator* mediator) : Colleague(mediator) {}
 
    void send(const std::string& message) override {
        mediator_->sendMessage(message, this);
    }
 
    void receive(const std::string& message) override {
        std::cout << "ConcreteColleagueA received: " << message << std::endl;
    }
};
 
// 具体同事类 B
class ConcreteColleagueB : public Colleague {
public:
    ConcreteColleagueB(Mediator* mediator) : Colleague(mediator) {}
 
    void send(const std::string& message) override {
        mediator_->sendMessage(message, this);
    }
 
    void receive(const std::string& message) override {
        std::cout << "ConcreteColleagueB received: " << message << std::endl;
    }
};
 
// 具体中介者类
class ConcreteMediator : public Mediator {
public:
    void addColleague(Colleague* colleague) {
        colleagues_.push_back(colleague);
    }
 
    void sendMessage(const std::string& message, Colleague* colleague) override {
        for (Colleague* col : colleagues_) {
            if (col != colleague) {
                col->receive(message);
            }
        }
    }
 
private:
    std::vector<Colleague*> colleagues_;
};
 
int main() {
    ConcreteMediator mediator;
 
    ConcreteColleagueA colleagueA(&mediator);
    ConcreteColleagueB colleagueB(&mediator);
 
    mediator.addColleague(&colleagueA);
    mediator.addColleague(&colleagueB);
 
    colleagueA.send("Hello from Colleague A");
    colleagueB.send("Hi from Colleague B");
 
    return 0;
}
18.备忘录模式（Memento Pattern）：
说明：在不破坏封装性的前提下，捕获一个对象的内部状态，并在该对象之外保存这个状态。这样以后就可以将该对象恢复到原先保存的状态。
应用：当要保存一个对象在某一个时刻的状态，以便在以后恢复到该状态时使用。
示例：
#include <iostream>
#include <string>
 
// 备忘录类
class Memento {
public:
    Memento(const std::string& state) : state_(state) {}
 
    std::string getState() const {
        return state_;
    }
 
private:
    std::string state_;
};
 
// 发起人类
class Originator {
public:
    void setState(const std::string& state) {
        state_ = state;
    }
 
    std::string getState() const {
        return state_;
    }
 
    Memento* createMemento() {
        return new Memento(state_);
    }
 
    void restoreMemento(Memento* memento) {
        state_ = memento->getState();
    }
 
private:
    std::string state_;
};
 
int main() {
    Originator originator;
 
    // 设置状态并创建备忘录
    originator.setState("State 1");
    Memento* memento = originator.createMemento();
 
    // 修改状态
    originator.setState("State 2");
 
    // 恢复到备忘录状态
    originator.restoreMemento(memento);
 
    std::cout << "Current state: " << originator.getState() << std::endl;
 
    delete memento;
 
    return 0;
}
19.观察者模式（Observer Pattern）：
说明：定义对象间的一种一对多的依赖关系，使得当一个对象状态发生改变时，所有依赖于它的对象都会得到通知并自动更新。
应用：当一个对象的改变需要同时改变其他对象时。
示例：
#include <iostream>
#include <vector>
 
// 抽象观察者
class Observer {
public:
    virtual void update(int value) = 0;
};
 
// 具体观察者 A
class ConcreteObserverA : public Observer {
public:
    void update(int value) override {
        std::cout << "ConcreteObserverA received update with value: " << value << std::endl;
    }
};
 
// 具体观察者 B
class ConcreteObserverB : public Observer {
public:
    void update(int value) override {
        std::cout << "ConcreteObserverB received update with value: " << value << std::endl;
    }
};
 
// 抽象主题
class Subject {
public:
    virtual void attach(Observer* observer) = 0;
    virtual void detach(Observer* observer) = 0;
    virtual void notify(int value) = 0;
};
 
// 具体主题
class ConcreteSubject : public Subject {
public:
    void attach(Observer* observer) override {
        observers_.push_back(observer);
    }
 
    void detach(Observer* observer) override {
        for (auto it = observers_.begin(); it != observers_.end(); ++it) {
            if (*it == observer) {
                observers_.erase(it);
                break;
            }
        }
    }
 
    void notify(int value) override {
        for (Observer* observer : observers_) {
            observer->update(value);
        }
    }
 
private:
    std::vector<Observer*> observers_;
};
 
int main() {
    ConcreteSubject subject;
 
    ConcreteObserverA observerA;
    ConcreteObserverB observerB;
 
    subject.attach(&observerA);
    subject.attach(&observerB);
 
    subject.notify(123);
 
    subject.detach(&observerA);
 
    subject.notify(456);
 
    return 0;
}
20.状态模式（State Pattern）：
说明：允许对象在其内部状态改变时改变它的行为，对象看起来似乎修改了它的类。
应用：当一个对象的行为取决于它的状态，并且它必须在运行时根据状态改变它的行为时。
示例：
#include <iostream>
 
// 抽象状态类
class State {
public:
    virtual void handle() = 0;
};
 
// 具体状态类 A
class ConcreteStateA : public State {
public:
    void handle() override {
        std::cout << "ConcreteStateA handled\n";
    }
};
 
// 具体状态类 B
class ConcreteStateB : public State {
public:
    void handle() override {
        std::cout << "ConcreteStateB handled\n";
    }
};
 
// 环境类
class Context {
public:
    Context(State* state) : state_(state) {}
 
    void setState(State* state) {
        state_ = state;
    }
 
    void request() {
        state_->handle();
    }
 
private:
    State* state_;
};
 
int main() {
    ConcreteStateA stateA;
    ConcreteStateB stateB;
 
    Context context(&stateA);
    context.request();
 
    context.setState(&stateB);
    context.request();
 
    return 0;
}
21.策略模式（Strategy Pattern）：
说明：定义一系列的算法，把它们一个个封装起来，并且使它们可以相互替换。使得算法可以独立于使用它的客户而变化。
应用：当有很多类，它们之间的区别仅在于它们的行为时。
示例：
#include <iostream>
 
// 抽象策略类
class Strategy {
public:
    virtual void execute() = 0;
};
 
// 具体策略类 A
class ConcreteStrategyA : public Strategy {
public:
    void execute() override {
        std::cout << "Executing strategy A\n";
    }
};
 
// 具体策略类 B
class ConcreteStrategyB : public Strategy {
public:
    void execute() override {
        std::cout << "Executing strategy B\n";
    }
};
 
// 上下文类
class Context {
public:
    Context(Strategy* strategy) : strategy_(strategy) {}
 
    void setStrategy(Strategy* strategy) {
        strategy_ = strategy;
    }
 
    void executeStrategy() {
        strategy_->execute();
    }
 
private:
    Strategy* strategy_;
};
 
int main() {
    ConcreteStrategyA strategyA;
    ConcreteStrategyB strategyB;
 
    Context context(&strategyA);
    context.executeStrategy();
 
    context.setStrategy(&strategyB);
    context.executeStrategy();
 
    return 0;
}
22.模板方法模式（Template Method Pattern）：
说明：定义一个操作中的算法的骨架，而将一些步骤延迟到子类中。模板方法使得子类可以不改变一个算法的结构即可重定义该算法的某些特定步骤。
应用：当要控制子类的扩展时，只允许特定的方法在特定的时间调用时。
示例：
#include <iostream>
 
// 抽象类
class AbstractClass {
public:
    void templateMethod() {
        primitiveOperation1();
        primitiveOperation2();
    }
 
    virtual void primitiveOperation1() = 0;
    virtual void primitiveOperation2() = 0;
};
 
// 具体类 A
class ConcreteClassA : public AbstractClass {
public:
    void primitiveOperation1() override {
        std::cout << "ConcreteClassA primitiveOperation1\n";
    }
 
    void primitiveOperation2() override {
        std::cout << "ConcreteClassA primitiveOperation2\n";
    }
};
 
// 具体类 B
class ConcreteClassB : public AbstractClass {
public:
    void primitiveOperation1() override {
        std::cout << "ConcreteClassB primitiveOperation1\n";
    }
 
    void primitiveOperation2() override {
        std::cout << "ConcreteClassB primitiveOperation2\n";
    }
};
 
int main() {
    AbstractClass* classA = new ConcreteClassA();
    classA->templateMethod();
 
    AbstractClass* classB = new ConcreteClassB();
    classB->templateMethod();
 
    delete classA;
    delete classB;
 
    return 0;
}
23.访问者模式（Visitor Pattern）：
说明：表示一个作用于某对象结构中的各元素的操作。它使你可以在不改变各元素类的前提下定义作用于这些元素的新操作。
应用：当要为一个对象的组合增加新的能力，而且封装并不重要时。
示例：
#include <iostream>
#include <vector>
 
// 前向声明
class ConcreteElementB;
 
// 抽象访问者
class Visitor {
public:
    virtual void visit(ConcreteElementB* element) = 0;
};
 
// 具体访问者 A
class ConcreteVisitorA : public Visitor {
public:
    void visit(ConcreteElementB* element) override {
        std::cout << "ConcreteVisitorA visited ConcreteElementB\n";
    }
};
 
// 具体访问者 B
class ConcreteVisitorB : public Visitor {
public:
    void visit(ConcreteElementB* element) override {
        std::cout << "ConcreteVisitorB visited ConcreteElementB\n";
    }
};
 
// 抽象元素
class Element {
public:
    virtual void accept(Visitor* visitor) = 0;
};
 
// 具体元素 A
class ConcreteElementA : public Element {
public:
    void accept(Visitor* visitor) override {
        // 具体元素 A 对访问者的反应
    }
};
 
// 具体元素 B
class ConcreteElementB : public Element {
public:
    void accept(Visitor* visitor) override {
        visitor->visit(this);
    }
};
 
// 对象结构
class ObjectStructure {
public:
    void attach(Element* element) {
        elements_.push_back(element);
    }
 
    void accept(Visitor* visitor) {
        for (Element* element : elements_) {
            element->accept(visitor);
        }
    }
 
private:
    std::vector<Element*> elements_;
};
 
int main() {
    ConcreteElementB elementB;
    ConcreteVisitorA visitorA;
    ConcreteVisitorB visitorB;
 
    elementB.accept(&visitorA);
    elementB.accept(&visitorB);
 
    return 0;
}
