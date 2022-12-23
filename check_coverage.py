import json

with open('coverage.json') as f:
    report = json.load(f)
    percent_covered = report['totals']['percent_covered']
    if percent_covered < 90:
        raise Exception('The total code coverage was less than 90%')
    else:
        print(f'The tests covered {percent_covered:.2f}% of the code.')

