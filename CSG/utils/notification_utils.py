"""
Taken from https://github.com/huggingface/knockknock/blob/master/knockknock/slack_sender.py
"""

from typing import List
import os
import datetime
import traceback
import functools
import json
import socket
import requests

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class SlackNotifier():

    def __init__(self, exp_name, config):
        self.exp_name = exp_name
        self.channel = config.CHANNEL
        self.enable = config.ENABLE
        self.webhook_url = config.WEBHOOK
        self.host_name = socket.gethostname()
        self.message_thread_id = ""
        self.dump = {
            "username": "Train-Notif",
            "channel": self.channel,
            "icon_emoji": ":clapper:",
        }
    def start_exp(self):
        if self.enable:
            self.start_time = datetime.datetime.now()
            dump = self.dump.copy()
            message = ['Training Started!',
                        'Experiment name: %s' % self.exp_name,
                        'Machine name: %s' % self.host_name,
                        'Starting date: %s' % self.start_time.strftime(DATE_FORMAT)]
            dump['text'] = '\n'.join(message)
            content = requests.post(self.webhook_url, json.dumps(dump))
            self.message_info = content
        # set the message id
    
    def log_info(self, info):
        """ Allow for generic updates on the thread (maybe per eval).
        """
        pass

    def exp_failed(self, ex):

        if self.enable:
            dump = self.dump.copy()
            end_time = datetime.datetime.now()
            elapsed_time = end_time - self.start_time
            self.start_time = datetime.datetime.now()
            dump = self.dump.copy()
            contents = ["Your training has crashed ☠️",
                        'Machine name: %s' % self.host_name,
                        'Experiment name: %s' % self.exp_name,
                        'Starting date: %s' % self.start_time.strftime(DATE_FORMAT),
                        'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                        'Crashed training duration: %s\n\n' % str(elapsed_time),
                        "Here's the error:",
                        '%s\n\n' % ex,
                        "Traceback:",
                        '%s' % traceback.format_exc()]
            dump['text'] = '\n'.join(contents)
            dump['icon_emoji'] = ':skull_and_crossbones:'
            content = requests.post(self.webhook_url, json.dumps(dump))


def slack_sender(webhook_url: str, channel: str, exp_name="", user_mentions: List[str] = []):
    """
    Slack sender wrapper: execute func, send a Slack notification with the end status
    (sucessfully finished or crashed) at the end. Also send a Slack notification before
    executing func.
    `webhook_url`: str
        The webhook URL to access your slack room.
        Visit https://api.slack.com/incoming-webhooks#create_a_webhook for more details.
    `channel`: str
        The slack room to log.
    `user_mentions`: List[str] (default=[])
        Optional users ids to notify.
        Visit https://api.slack.com/methods/users.identity for more details.
    """

    dump = {
        "username": "Train-Notif",
        "channel": channel,
        "icon_emoji": ":clapper:",
    }
    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):

            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += ' - RANK: %s' % os.environ['RANK']
            else:
                master_process = True

            if master_process:
                contents = ['Your training has started 🎬',
                            'Experiment name: %s' % exp_name,
                            'Machine name: %s' % host_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(DATE_FORMAT)]
                contents.append(' '.join(user_mentions))
                dump['text'] = '\n'.join(contents)
                dump['icon_emoji'] = ':clapper:'
                requests.post(webhook_url, json.dumps(dump))

            try:
                value = func(*args, **kwargs)

                if master_process:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ["Your training is complete 🎉",
                                'Experiment name: %s' % exp_name,
                                'Machine name: %s' % host_name,
                                'Main call: %s' % func_name,
                                'Starting date: %s' % start_time.strftime(DATE_FORMAT),
                                'End date: %s' % end_time.strftime(DATE_FORMAT),
                                'Training duration: %s' % str(elapsed_time)]

                    try:
                        str_value = str(value)
                        contents.append('Main call returned value: %s'% str_value)
                    except:
                        contents.append('Main call returned value: %s'% "ERROR - Couldn't str the returned value.")

                    contents.append(' '.join(user_mentions))
                    dump['text'] = '\n'.join(contents)
                    dump['icon_emoji'] = ':tada:'
                    requests.post(webhook_url, json.dumps(dump))

                return value

            except Exception as ex:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                contents = ["Your training has crashed ☠️",
                            'Machine name: %s' % host_name,
                            'Experiment name: %s' % exp_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(DATE_FORMAT),
                            'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                            'Crashed training duration: %s\n\n' % str(elapsed_time),
                            "Here's the error:",
                            '%s\n\n' % ex,
                            "Traceback:",
                            '%s' % traceback.format_exc()]
                contents.append(' '.join(user_mentions))
                dump['text'] = '\n'.join(contents)
                dump['icon_emoji'] = ':skull_and_crossbones:'
                requests.post(webhook_url, json.dumps(dump))
                raise ex

        return wrapper_sender

    return decorator_sender