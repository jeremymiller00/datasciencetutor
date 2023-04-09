from langchain.llms import OpenAI

# llm = OpenAI(temperature=0.9)
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)


def ask(question):
    print ("\nYou asked:")
    print(question)
    return llm(question)

def main():
    while True:
        print("\n<<<>>>")
        print("Type (exit) to exit\n")
        question = input("How can I help?\n\n")
        if question == "(exit)":
            return
        else:
            print(ask(question))

######################
if __name__ == '__main__':
    main()