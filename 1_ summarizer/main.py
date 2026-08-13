from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain")

    information = """
        # Elon Musk

        ## Overview

        Elon Reeve Musk (born June 28, 1971, in Pretoria, South Africa) is a
        businessman and former public official. He is the CEO and largest
        shareholder of Tesla and SpaceX.

        ## Personal details

        - Full name: Elon Reeve Musk
        - Born: June 28, 1971, in Pretoria, South Africa
        - Citizenship: South Africa (since 1971), Canada (since 1971), and the
          United States (since 2002)
        - Political party: Independent
        - Education: University of Pennsylvania (BA, BS)
        - Parents: Errol Musk and Maye Musk
        - Spouses: Justine Wilson (2000–2008) and Talulah Riley (2010–2012 and
          2013–2016)
        - Children: 14, including Vivian Wilson

        ## Public service

        Musk served as Senior Advisor to the President for Government Efficiency
        from January 20 to May 30, 2025. He served under President Donald Trump
        alongside Massad Boulos.

        ## Career and organizations

        - Tesla: CEO and product architect
        - SpaceX: Founder, CEO, and chief engineer
        - xAI: Founder and CEO
        - X Corp.: Founder and CTO
        - Musk Foundation: President
        - Other ventures: Co-founder of Neuralink, The Boring Company, OpenAI,
          Zip2, X.com, and PayPal

        ## Career history

        Musk moved to Canada in 1989 and earned bachelor's degrees from the
        University of Pennsylvania in 1997. He co-founded Zip2 in 1995. After its
        sale in 1999, he co-founded X.com, which merged with Confinity in 2000 to
        form PayPal. eBay acquired PayPal in 2002.

        In 2002, Musk founded SpaceX. He joined Tesla as an early investor in 2004
        and became its CEO and product architect in 2008. He co-founded OpenAI in
        2015 and later founded xAI. He acquired Twitter in 2022 and renamed it X
        in 2023. His other businesses include Neuralink and The Boring Company.

        ## Wealth and compensation

        Musk has been described as the world's wealthiest person since 2025. As
        of August 8, 2026, Forbes estimated his net worth at US$823 billion. In
        November 2025, Tesla approved a performance-based compensation package
        worth up to $1 trillion over 10 years if specified milestones are met.
        """

    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
