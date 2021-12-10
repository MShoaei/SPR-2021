import click
import homework


@click.group()
def cli():
    pass


cli.add_command(homework.hw1)
cli.add_command(homework.hw2)
cli.add_command(homework.hw3)

if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = [10, 5]
    cli()
